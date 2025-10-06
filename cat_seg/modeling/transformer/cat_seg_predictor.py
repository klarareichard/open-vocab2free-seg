# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch
import random

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates

import numpy as np
import open_clip
import spacy
class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        vocab_free: str = False,
        all_adjectives: str = False,
        adjectives: str = False,
        weighted_sampling: str = False,
        agg_strategy: str = 'union',
        seed: int = 8262,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates
        self.nlp = spacy.load("en_core_web_sm")
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer
        
        self.tokens = None
        self.cache = None
        self.vocab_free = vocab_free
        self.all_adjectives = all_adjectives
        self.adjectives = adjectives
        self.agg_strategy = agg_strategy
        self.weighted_sampling = weighted_sampling

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        ret["vocab_free"] = cfg.VOCAB_FREE
        ret["all_adjectives"] = cfg.ALL_ADJECTIVES
        ret["adjectives"] = cfg.ADJECTIVES
        ret["agg_strategy"] = cfg.AGG_STRATEGY
        ret["weighted_sampling"] = cfg.WEIGHTED_SAMPLING

        ret["seed"] = cfg.MODEL.SEED

        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None, adjectives=None, mapped_class_names = None, predicted_class_names = None, clusters = None):
        #if not self.adjectives: 
        #    adjectives = None
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt, adjectives, mapped_class_names, predicted_class_names)
        text = text.repeat(x.shape[0], 1, 1, 1) if x.shape[0] != text.shape[0] else text
        text = text[:, gt_cls, :, :] if gt_cls is not None else text
        out = self.transformer(x, text, vis)

        if gt_cls is not None:
            C_all = self.class_texts if self.training else self.test_class_texts
            C_all = len(C_all)

            B, C, H, W = out.size()
            out_all = torch.zeros(B, C_all, H, W, dtype=out.dtype, device=out.device, requires_grad=out.requires_grad)

            for i, c in enumerate(gt_cls):
                out_all = out_all.index_add(1, c, out[:, i, :, :].unsqueeze(1))
            return out_all

        # Apply clustering aggregation if vocabulary free
        if self.vocab_free and clusters is not None and self.agg_strategy not in ['early', 'late']:
            # DO NOT USE
            out = self.aggregate_cost_maps(out, clusters[0], strategy=self.agg_strategy)

        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts)#,context_length=10, truncate=True)
                texts= texts.cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    

    def classify_attributes_with_spacy(self, attribute_list):
        before_noun = []
        after_noun = []

        for attribute in attribute_list:
            doc = self.nlp(attribute)

            # Check the first word's part-of-speech tag and dependency
            first_token = doc[0]
            if first_token.dep_ in {'prep', 'aux'} or first_token.pos_ in {'VERB'}:
                # If it starts with a preposition, auxiliary verb, or main verb
                after_noun.append(attribute)
            else:
                # Otherwise, assume it comes before the noun
                before_noun.append(attribute)

        return before_noun, after_noun
    
    def adapt_adjectives(self, mapped_class_names, class_names, adjectives):
        # Check if mapped_class_names is not None
        if mapped_class_names is not None:
            # Create a new adjectives dictionary with mapped class names as keys
            new_adjectives = {}

            # Iterate through the class_names and mapped_class_names together
            for original_name, mapped_name in zip(class_names, mapped_class_names):
                if original_name in adjectives:
                    new_adjectives[mapped_name] = adjectives[original_name]

            # Return the new adjectives dictionary with updated keys
            return new_adjectives

        # If mapped_class_names is None, return the original adjectives dictionary
        return adjectives

    def compute_all_adjective_weights(self, all_adjectives):
        # Count all adjective occurrences
        adjective_counts = {}
        for class_adjectives in all_adjectives.values():
            for adj in class_adjectives:
                adjective_counts[adj] = adjective_counts.get(adj, 0) + 1

        # Compute weight for each adjective
        return {adj: 1 / (count + 0.1) for adj, count in adjective_counts.items()}

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None, adjectives=None, mapped_class_names=None,
                        predicted_class_names=None):
        print(adjectives)
        #if not self.adjectives: 
        #    adjectives = None
        #print(adjectives)
        B = len(adjectives) if adjectives is not None else 1

        """
        if adjectives is not None and mapped_class_names is not None and predicted_class_names is not None:
            for i in range(B):
                adjectives[i] = self.adapt_adjectives(mapped_class_names[i], predicted_class_names[i], adjectives[i])
                predicted_class_names[i] = mapped_class_names[i] if mapped_class_names[i] is not None else predicted_class_names[i]
        """
        if self.vocab_free:
            classnames = predicted_class_names[0]# [s.strip("'") for s in predicted_class_names[0].strip("[]").split(", ")]
            # classnames = [s.strip("'") for s in predicted_class_names[0].strip("[]").split(", ")]
            #classnames = [*{*classnames}]
            """
            classnames = [x for x in adjectives[0].keys()] # TODO: valid only for inference or num_gpus == batch_size
            if classnames == []:
                a = 1
            """

        C = len(classnames)

        all_tokens = []
        for i in range(B):
            batch_tokens = []
            adjective_weights = self.compute_all_adjective_weights(adjectives[i]) if adjectives is not None and adjectives[i] is not None and self.weighted_sampling else None
            for classname in classnames:
                adj_desc_before, adj_desc_after = None, None
                if adjectives is not None and adjectives[i] is not None and classname in adjectives[i] and self.adjectives:
                    adjectives_per_class = adjectives[i][classname]
                    if adjectives_per_class:
                        if self.all_adjectives:
                            # Accumulate all adjectives if self.all_adjectives is True
                            adj_desc_before, adj_desc_after = [], []
                            for adjective in adjectives_per_class:
                                attribute_list = [adjective]
                                before_noun, after_noun = self.classify_attributes_with_spacy(attribute_list)
                                if before_noun:
                                    adj_desc_before.extend(before_noun)
                                if after_noun:
                                    adj_desc_after.extend(after_noun)
                        else:
                            # Select a random adjective if self.all_adjectives is False
                            if self.weighted_sampling:
                                weights = [adjective_weights[adj] for adj in adjectives_per_class]
                                adjective = random.choices(adjectives_per_class, weights=weights, k=1)[0]
                            else:
                                adjective = random.choice(adjectives_per_class)
                            attribute_list = [adjective]
                            print(attribute_list)
                            before_noun, after_noun = self.classify_attributes_with_spacy(attribute_list)
                            adj_desc_before = " ".join(before_noun) if before_noun else None
                            adj_desc_after = " ".join(after_noun) if after_noun else None

                formatted_text = classname

                # Helper function to join adjectives with 'and' before the last one
                def join_adjectives(adj_list):
                    if len(adj_list) > 1:
                        return ', '.join(adj_list[:-1]) + ' and ' + adj_list[-1]
                    elif len(adj_list) == 1:
                        return adj_list[0]
                    return ''

                # Join all adjectives if self.all_adjectives is True
                if self.all_adjectives:
                    if adj_desc_before:
                        formatted_text = f"{join_adjectives(adj_desc_before)} {formatted_text}"  # Join adjectives before the class name
                    if adj_desc_after:
                        formatted_text = f"{formatted_text} {join_adjectives(adj_desc_after)}"  # Join adjectives after the class name
                else:  # Current behavior if self.all_adjectives is False
                    if adj_desc_before:
                        formatted_text = f"{adj_desc_before} {formatted_text}"
                    if adj_desc_after:
                        formatted_text = f"{formatted_text} {adj_desc_after}"


                texts = [template.format(formatted_text) for template in templates]
                #print(texts)
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else:
                    texts = clip.tokenize(texts).cuda()
                batch_tokens.append(texts)
            all_tokens.append(torch.stack(batch_tokens, dim=0).squeeze(1))

        if B > 1:
            tokens = torch.stack(all_tokens, dim=0)
        else:
            tokens = all_tokens[0]

        if adjectives is None or B == 1:
            class_embeddings = clip_model.encode_text(tokens, prompt)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            return class_embeddings.unsqueeze(1)
        else:
            # Reshape tokens to (B*C, tokenizer_size)
            tokens_reshaped = tokens.view(-1, tokens.size(-1))
            class_embeddings = clip_model.encode_text(tokens_reshaped, prompt)
            # Reshape back to (B, C, embed_dim)
            class_embeddings = class_embeddings.view(B, C, -1)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            return class_embeddings.unsqueeze(2)

    def aggregate_cost_maps(self, out, clusters, strategy='union', overlap_threshold=0.5, temperature=1.0):
        """
        Aggregate cost maps according to cluster information using different strategies.

        Args:
            out (torch.Tensor): Output tensor of shape (B, C, H, W) containing cost maps
            clusters (dict): Dictionary mapping cluster IDs to lists of class names/indices
            strategy (str): Aggregation strategy ('union', 'max', 'weighted')
            overlap_threshold (float): Threshold for considering maps as overlapping (0-1)
            temperature (float): Temperature parameter for softmax weighting

        Returns:
            torch.Tensor: Aggregated cost maps
        """

        B, C, H, W = out.size()

        # Create a mapping from class index to cluster index
        class_to_cluster = {}
        for cluster_idx, classes in clusters.items():
            for class_name in classes:
                class_to_cluster[class_name] = cluster_idx

        # Initialize output tensor with same size as input but with number of clusters
        num_clusters = len(clusters)
        aggregated_maps = torch.zeros(B, num_clusters, H, W, device=out.device)

        if strategy == 'union':
            # Simple union (sum) of cost maps within each cluster
            for cluster_idx, classes in clusters.items():
                class_indices = [i for i, c in enumerate(classes)]
                cluster_maps = out[:, class_indices]
                aggregated_maps[:, int(cluster_idx)-1] = cluster_maps.sum(dim=1)

        elif strategy == 'max':
            # Take maximum response within each cluster
            for cluster_idx, classes in clusters.items():
                class_indices = [i for i, c in enumerate(classes)]
                cluster_maps = out[:, class_indices]
                aggregated_maps[:, int(cluster_idx)-1] = cluster_maps.max(dim=1)[0]

        elif strategy == 'weighted':
            # Weighted combination based on spatial overlap
            for cluster_idx, classes in clusters.items():
                class_indices = [i for i, c in enumerate(classes)]
                cluster_maps = out[:, class_indices]

                # Compute pairwise overlap between maps in the cluster
                B, N, H, W = cluster_maps.size()
                weights = torch.ones(B, N, device=out.device)

                for i in range(N):
                    for j in range(i + 1, N):
                        map1 = cluster_maps[:, i]
                        map2 = cluster_maps[:, j]

                        # Compute normalized overlap
                        intersection = torch.min(map1, map2).sum(dim=(1, 2))
                        union = torch.max(map1, map2).sum(dim=(1, 2))
                        overlap = intersection / (union + 1e-6)

                        # Update weights based on overlap
                        mask = overlap > overlap_threshold
                        weights[:, i] += mask.float()
                        weights[:, j] += mask.float()

                # Apply softmax to get normalized weights
                weights = F.softmax(weights / temperature, dim=1)

                # Weighted sum of maps
                weighted_maps = cluster_maps * weights.view(B, N, 1, 1)
                aggregated_maps[:, int(cluster_idx)-1] = weighted_maps.sum(dim=1)

        return aggregated_maps