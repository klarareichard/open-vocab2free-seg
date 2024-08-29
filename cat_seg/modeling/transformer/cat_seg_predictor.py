# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

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
            
        print(self.test_class_texts)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #print(self.test_class_texts)
  
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

        #self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        #self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
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

        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None, adjectives=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        #text = [text[c] for c in gt_cls] if gt_cls is not None else text
        #text = text[gt_cls] if gt_cls is not None else text
        
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt, adjectives)
        text = text[gt_cls] if gt_cls is not None else text
        text = text.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x, text, vis)
        
        if gt_cls is not None:
            C_all = self.class_texts if self.training else self.test_class_texts
            C_all = len(C_all)

            B, C, H, W = out.size()
            out_all = torch.zeros(B, C_all, H, W, dtype=out.dtype, device=out.device, requires_grad=out.requires_grad)

            for i, c in enumerate(gt_cls):
                out_all = out_all.index_add(1, c, out[:, i, :, :].unsqueeze(1))
            return out_all

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
                #print(texts)
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                #print("tokenizer")
                texts = self.tokenizer(texts).cuda()
            else: 
                #print("else")
                texts = clip.tokenize(texts)#,context_length=10, truncate=True)
                
                print(texts.shape)
                print("else")
                #print("text shape")
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
    

    def classify_attributes_with_spacy(attribute_list):
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
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None, adjectives=None):

        tokens = []
        # Process each classname
        for classname in classnames:
            if classname in adjectives:
                adjectives_per_class = adjectives[classname]
                if len(adjectives_per_class):
                    adjective = adjectives_per_class[0]
                    print(adjective)
                before_noun, after_noun = self.classify_attributes_with_spacy([adjective])

                print("BEFORE NOUN")
                print(before_noun)
                adj_desc_after = None
                adj_desc_before = None
                if len(before_noun):
                    adj_desc_before = " ".join(before_noun)
                if len(after_noun):
                    adj_desc_after = " ".join(after_noun)
                #print(adj_desc)
            else:
                adj_desc_before = None
                adj_desc_after = None
            
            
            #formatted_text = f"{adj_desc_before} {classname}" if adj_desc else classname
            formatted_text = classname

            # Add adj_desc_before if it is not None
            if adj_desc_before:
                formatted_text = f"{adj_desc_before} {formatted_text}"

            # Add adj_desc_after if it is not None
            if adj_desc_after:
                formatted_text = f"{formatted_text} {adj_desc_after}"
            texts = [template.format(formatted_text) for template in templates]
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            tokens.append(texts)
        tokens = torch.stack(tokens, dim=0).squeeze(1)

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        class_embeddings = class_embeddings.unsqueeze(1)

        return class_embeddings