# Adapted from https://github.com/altndrr/vicss/

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassRecall
from typing import List, Tuple
import json
from collections import OrderedDict


from .singletons import SentenceBERT


class SemanticJaccardIndex:
    """Semantic jaccard index metric factory.

    Args:
        mode (str): Mode to use. Either "hard", "soft", or "overlap", "nearest".
    """

    def __new__(cls, mode: str, *args, **kwargs) -> Metric:
        if mode == "hard":
            return SemanticHardJaccardIndex(*args, **kwargs)
        elif mode == "soft":
            return SemanticSoftJaccardIndex(*args, **kwargs)
        elif mode == "overlap":
            return SemanticClusterJaccardIndex(*args, match="overlap", **kwargs)
        elif mode == "nearest":
            return SemanticClusterJaccardIndex(*args, match="nearest", **kwargs)
        raise ValueError(f"Invalid mode {mode}")


class SemanticRecall:
    """Semantic recall metric factory.

    Args:
        mode (str): Mode to use. Either "hard", "soft", or "overlap", "nearest".
    """

    def __new__(cls, mode: str, *args, **kwargs) -> Metric:
        if mode == "hard":
            return SemanticHardRecall(*args, **kwargs)
        elif mode == "soft":
            return SemanticSoftRecall(*args, **kwargs)
        raise ValueError(f"Invalid mode {mode}")

class SemanticClusterJaccardIndex(Metric):
    def __init__(
            self, *args, classes: List[str], average: str = "micro", match: str = "overlap", ignore_index=255, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        assert match in ["overlap", "nearest"]
        self.average = average
        self.match = match
        self.classes = classes
        self.ignore_index = ignore_index
        self.encoder = SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: List[Tuple[list, np.ndarray]], targets: List[np.ndarray]) -> None:
        intersections, unions = [], []
        target_idxs = []

        classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

        for value, target in zip(values, targets):
            names, value_mask = value
            target = torch.tensor(target, device=self.device)
            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device)

            # Ensure value_mask has the same spatial dimensions as target
            if value_mask.shape[-2:] != target.shape[-2:]:
                value_mask = F.interpolate(
                    value_mask.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)
            value_mask = value_mask.argmax(dim=0)

            cls_idx = torch.unique(target).long()
            cls_idx = cls_idx[cls_idx != self.ignore_index]  # Remove ignore index if present

            # Assign predicted classes to target classes
            if self.match == "overlap":
                matrix_size = (value_mask.max() + 1, len(self.classes))
                v, t = value_mask.view(-1), target.view(-1)
                # Filter out ignore_index values
                valid_mask = t != self.ignore_index
                v, t = v[valid_mask], t[valid_mask]
                co_occurrences = torch.zeros(matrix_size, dtype=torch.long, device=self.device)
                co_occurrences = torch.bincount(
                    v * matrix_size[1] + t, minlength=matrix_size[0] * matrix_size[1]
                ).view(matrix_size)
                value_mask[target != self.ignore_index] = co_occurrences.argmax(dim=-1)[value_mask[target != self.ignore_index]]
            elif self.match == "nearest":
                # Compute the text similarity between the predictions and the labels
                value_names_z = self.encoder(names).to(self.device)
                value_names_z = value_names_z if value_names_z.dim() == 2 else value_names_z.unsqueeze(0)
                value_names_z = value_names_z.unsqueeze(1)
                similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))
                similarity[:, ~cls_idx] = 0
                value_mask = similarity.argmax(dim=-1)[value_mask]

            matches = (value_mask == target).long()
            for idx in cls_idx:
                mask = target == idx

                intersection = torch.sum(matches[mask])
                union = torch.sum(matches) + torch.sum(mask) - intersection

                intersections.append(intersection)
                unions.append(union)
                target_idxs.append(idx)

        intersections = torch.stack(intersections)
        unions = torch.stack(unions)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        if self.average == "micro":
            return torch.mean(self.intersection.float() / (self.union.float() + 1e-8))
        elif self.average == "macro":
            jaccard_indexes = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                class_intersection = self.intersection[mask].float()
                class_union = self.union[mask].float()
                jaccard_indexes.append(torch.mean(class_intersection / (class_union + 1e-8)))

            return torch.mean(torch.stack(jaccard_indexes))


class SemanticWeightedJaccardIndex:
    def __init__(self, dataset_json, ignore_index, similarity_matrix, device="cpu"):
        """
        Initializes the SemanticWeightedJaccardIndex for calculating weighted mean IoU.

        Args:
        - dataset_json (str): Path to JSON file containing the list of class names for target masks.
        - ignore_index (int): Index to ignore in the IoU calculation.
        - similarity_matrix (dict): Similarity matrix with class names as keys.
        - device (str): Device to run the computation on.
        """
        # Load the class names for the target masks
        with open(dataset_json, 'r') as file:
            self.target_class_names = json.load(file)
        
        self.ignore_index = ignore_index
        self.similarity_matrix = similarity_matrix
        self.device = device
        self.reset()

    def reset(self):
        """Resets the accumulated values."""
        self.total_weighted_similarity = 0.0
        self.total_pixels = 0
        
    import torch
    import torch.nn.functional as F

#    def update(self, pred_masks, gt_masks):
#        for (pred_class_names, pred_mask), gt_mask in zip(pred_masks, gt_masks):
            # Convert predicted mask from (num_classes, height, width) to (height, width)
#            value_mask = torch.tensor(pred_mask, device=self.device, dtype=torch.float) #.unsqueeze(0)

            # Ensure value_mask matches target shape
            #if value_mask.shape[-2:] != gt_mask.shape[-2:]:
            #    value_mask = F.interpolate(
            #        value_mask.unsqueeze(0), size=gt_mask.shape[-2:], mode="bilinear", align_corners=False
            #    ).squeeze(0)

            # Convert the interpolated mask to a single-channel mask
#            value_mask = value_mask.argmax(dim=0)  # (height, width)

            # Create class-to-index mappings
#            pred_name_to_index = {name: idx for idx, name in enumerate(pred_class_names)}
#            gt_name_to_index = {name: idx for idx, name in enumerate(self.target_class_names)}

            # Convert gt_mask to tensor and flatten
#            gt_flat = torch.tensor(gt_mask, device=self.device).view(-1)  # Flatten ground truth mask

            # Generate valid mask for ground truth
#            valid_mask = gt_flat != self.ignore_index

            # Print shapes for debugging
#            print(f"value_mask shape: {value_mask.shape}, gt_flat shape: {gt_flat.shape}, valid_mask shape: {valid_mask.shape}")
            
            

            # Flatten the value_mask
#            pred_flat = value_mask.view(-1).cpu()  # Flatten and move to CPU
            
#            gt_flat = gt_flat.cpu()  # Move ground truth to CPU

            # Validate length of filtered arrays
#            valid_mask_cpu = valid_mask.cpu()  # Ensure valid_mask is also on CPU

#            if valid_mask_cpu.sum() > 0:
 #               valid_gt_flat = gt_flat[valid_mask_cpu]  # Ground truth pixels that are valid
#                valid_pred_flat = pred_flat[valid_mask_cpu]  # Predicted pixels that are valid

                # Check shapes before proceeding
#                print(f"valid_gt_flat shape: {valid_gt_flat.shape}, valid_pred_flat shape: {valid_pred_flat.shape}")

#                if valid_gt_flat.shape[0] != valid_pred_flat.shape[0]:
#                    print(f"Shape mismatch: gt {valid_gt_flat.shape[0]}, pred {valid_pred_flat.shape[0]}")
#                    continue  # Skip this pair or handle accordingly

                # Calculate similarity for valid pixels
#                for gt_val, pred_val in zip(valid_gt_flat, valid_pred_flat):
#                    gt_class_name = self.target_class_names[gt_val.item()]
#                    pred_class_name = pred_class_names[pred_val.item()]

#                    similarity = self.similarity_matrix.get(gt_class_name, {}).get(pred_class_name, 0.0)
#                    self.total_weighted_similarity += similarity

                # Update total pixel count
#                self.total_pixels += valid_mask.sum().item()
#            else:
#                print("No valid pixels found, skipping this mask pair.")




            

#    def compute(self):
#        """Computes the final weighted mean IoU."""
#        if self.total_pixels == 0:
#            return 0.0  # Avoid division by zero if no valid pixels are available
#        return self.total_weighted_similarity / self.total_pixels

    
    def update(self, pred_masks, gt_masks):
        for (pred_class_names, pred_mask), gt_mask in zip(pred_masks, gt_masks):
            # Convert predicted mask from (num_classes, height, width) to (height, width)
            value_mask = torch.tensor(pred_mask, device=self.device, dtype=torch.float)

            # Convert to single-channel mask
            value_mask = value_mask.argmax(dim=0)  # (height, width)

            # Create class-to-index mappings
            pred_name_to_index = {name: idx for idx, name in enumerate(pred_class_names)}
            gt_name_to_index = {name: idx for idx, name in enumerate(self.target_class_names)}

            # Convert gt_mask to tensor and flatten
            gt_flat = torch.tensor(gt_mask, device=self.device).view(-1)  # Flatten ground truth mask

            # Generate valid mask for ground truth
            valid_mask = gt_flat != self.ignore_index  # Shape: (349696,)

            # Print shapes for debugging
            print(f"value_mask shape: {value_mask.shape}, gt_flat shape: {gt_flat.shape}, valid_mask shape: {valid_mask.shape}")

            # Flatten the value_mask
            pred_flat = value_mask.view(-1)  # Keep on GPU

            # Validate length of filtered arrays
            if valid_mask.sum() > 0:
                valid_gt_flat = gt_flat[valid_mask]  # Ground truth pixels that are valid
                valid_pred_flat = pred_flat[valid_mask]  # Predicted pixels that are valid

                # Check shapes before proceeding
                print(f"valid_gt_flat shape: {valid_gt_flat.shape}, valid_pred_flat shape: {valid_pred_flat.shape}")

                if valid_gt_flat.shape[0] != valid_pred_flat.shape[0]:
                    print(f"Shape mismatch: gt {valid_gt_flat.shape[0]}, pred {valid_pred_flat.shape[0]}")
                    continue  # Skip this pair or handle accordingly

                # Prepare indices for ground truth and predicted classes
                gt_indices = valid_gt_flat.long()  # Ensure indices are long
                pred_indices = valid_pred_flat.long()  # Ensure indices are long

                # Convert indices to class names
                gt_class_names = [self.target_class_names[idx] for idx in gt_indices.cpu().numpy()]
                pred_class_names = [pred_class_names[idx] for idx in pred_indices.cpu().numpy()]

                # Retrieve similarity scores for all valid pairs
                similarities = torch.tensor([
                    self.similarity_matrix.get(gt_name, {}).get(pred_name, 0.0) 
                    for gt_name, pred_name in zip(gt_class_names, pred_class_names)
                ], device=self.device)

                # Sum the similarities
                self.total_weighted_similarity += similarities.sum().item()

                # Update total pixel count
                self.total_pixels += valid_mask.sum().item()
            else:
                print("No valid pixels found, skipping this mask pair.")

    def compute(self):
        """Computes the final weighted mean IoU."""
        if self.total_pixels == 0:  # Use item() to get float for comparison
            return 0.0
        
        return self.total_weighted_similarity / self.total_pixels  # Convert to float for final division




class SemanticHardJaccardIndex(MulticlassJaccardIndex):
    """Metric to evaluate the semantic Jaccard index.

    Extends the original `torchmetrics.classification.MulticlassJaccardIndex` to support
    semantic masks composed of a list of class names and a corresponding semantic mask.

    Args:
        classes (list[str]): List of class names.
    """

    def __init__(self, *args, classes: List[str], ignore_index=255, **kwargs) -> None:
        super().__init__(*args, num_classes=len(classes), **kwargs)
        self.classes = classes
        self.class_to_idx = defaultdict(lambda: 0)
        self.class_to_idx.update({c: i for i, c in enumerate(classes)})
        self.ignore_index = ignore_index

    def update(self, values: List[Tuple[list, np.ndarray]], targets: List[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        for value, target in zip(values, targets):
            names, value_mask = value
            value_mask = torch.tensor(value_mask, dtype=torch.float).unsqueeze(0)
            if value_mask.shape[-2:] != target.shape[-2:]:
                value_mask = F.interpolate(
                    value_mask.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)
            value_mask = value_mask.argmax(dim=1).squeeze(0).numpy()

            values_idx_to_class_idx = defaultdict(lambda: len(self.classes))
            values_idx_to_class_idx.update({i: self.class_to_idx[v] for i, v in enumerate(names)})
            value_mask = np.vectorize(values_idx_to_class_idx.get)(value_mask)

            value = torch.tensor(value_mask, device=self.device)
            target = torch.tensor(target.astype(np.int32), device=self.device)


            super().update(value, target)


class SemanticSoftJaccardIndex(Metric):
    def __init__(self, *args, classes: List[str], average: str = "micro", ignore_index=255, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        self.classes = classes
        self.ignore_index = ignore_index
        self.average = average
        self.encoder = SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: List[Tuple[list, np.ndarray]], targets: List[np.ndarray]) -> None:
        intersections, unions = [], []
        target_idxs = []

        classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

        for value, target in zip(values, targets):
            names, value_mask = value
            target = torch.tensor(target, device=self.device)
            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device).unsqueeze(0)

            # Ensure value_mask has the same spatial dimensions as target
            if value_mask.shape[-2:] != target.shape[-2:]:
                value_mask = F.interpolate(
                    value_mask.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)
            value_mask = value_mask.argmax(dim=1).squeeze(0)

            # Compute the text similarity between the predictions and the labels
            value_names_z = self.encoder(names).to(self.device)
            value_names_z = value_names_z if value_names_z.dim() == 2 else value_names_z.unsqueeze(0)
            value_names_z = value_names_z.unsqueeze(1)
            similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))

            # Extract the class indexes
            cls_idx = torch.unique(target).long()
            cls_idx = cls_idx[cls_idx != self.ignore_index]  # Remove ignore index if present

            value_scores = similarity[:, cls_idx][value_mask.long()].permute(2, 0, 1)

            for i, idx in enumerate(cls_idx):

                mask = target == idx
                non_mask = target != self.ignore_index

                intersection = torch.sum(value_scores[i][mask * non_mask])
                union = torch.sum(value_scores[i][non_mask]) + torch.sum(mask) - intersection

                intersections.append(intersection)
                unions.append(union)
                target_idxs.append(idx)

        intersections = torch.tensor(intersections, device=self.device)
        unions = torch.tensor(unions, device=self.device)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        if self.average == "micro":
            return torch.mean(self.intersection.float() / self.union.float())
        elif self.average == "macro":
            jaccard_indexes = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                class_intersection = self.intersection[mask].float()
                class_union = self.union[mask].float()
                jaccard_indexes.append(torch.mean(class_intersection / class_union))

            return torch.mean(torch.stack(jaccard_indexes))


class SemanticHardRecall(MulticlassRecall):
    """Metric to evaluate the semantic recall score.

    Extends the original `torchmetrics.classification.MulticlassRecall` to support semantic masks
    composed of a list of class names and a corresponding semantic mask.

    Args:
        classes (list[str]): List of class names.
    """

    def __init__(self, *args, classes: List[str], ignore_index=255, **kwargs) -> None:
        super().__init__(*args, num_classes=len(classes), **kwargs)
        self.classes = classes
        self.class_to_idx = defaultdict(lambda: 0)
        self.class_to_idx.update({c: i for i, c in enumerate(classes)})
        self.ignore_index = ignore_index

    def update(self, values: List[Tuple[list, np.ndarray]], targets: List[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        for value, target in zip(values, targets):
            names, value_mask = value

            value_mask = torch.tensor(value_mask, dtype=torch.float).unsqueeze(0)
            value_mask = F.interpolate(
                value_mask, size=target.shape, mode="bilinear", align_corners=False
            )
            value_mask = value_mask.argmax(dim=1).squeeze(0).numpy()

            values_idx_to_class_idx = defaultdict(lambda: len(self.classes))
            values_idx_to_class_idx.update({i: self.class_to_idx[v] for i, v in enumerate(names)})
            value_mask = np.vectorize(values_idx_to_class_idx.get)(value_mask)

            value = torch.tensor(value_mask, device=self.device)
            target = torch.tensor(target.astype(np.int32), device=self.device)


            super().update(value, target)


class SemanticSoftRecall(Metric):
    """Metric to evaluate the semantic soft recall score.

    It takes as input semantic masks composed of a list of class names and a corresponding
    semantic mask. Since predictions and class names may differ, the metric computes the
    intersection and union between the predicted semantic masks and the target semantic masks
    by considering scores instead of binary values. The scores are computed as the cosine
    similarity between the predicted semantic mask and the semantic mask of each class.

    Args:
        classes (list[str]): List of class names.
        average (str): Type of averaging to perform. Can be "micro" or "macro". Defaults to
            "micro".
    """

    def __init__(self, *args, classes: List[str], average: str = "micro", ignore_index=255, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert average in ["micro", "macro"]
        self.classes = classes
        self.ignore_index = ignore_index
        self.average = average
        self.encoder = SentenceBERT()
        classes_z = self.encoder(classes)
        self.encoder.register_buffer("classes_z", classes_z, exists_ok=True)

        self.add_state("recall", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("target_idx", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: List[Tuple[list, np.ndarray]], targets: List[np.ndarray]) -> None:
        """Update state with data.

        Args:
            values (list[tuple[list, np.ndarray]]): Predicted semantic masks. The first element
                of the tuple is a list of class names, while the second element is the predicted
                semantic mask.
            targets (list[np.ndarray]): Targets masks.
        """
        recall = []
        target_idxs = []

        classes_z = self.encoder.classes_z.unsqueeze(0).to(self.device)

        for value, target in zip(values, targets):
            names, value_mask = value
            target = torch.tensor(target.astype(np.int32), device=self.device)

            value_mask = torch.tensor(value_mask, dtype=torch.float, device=self.device)

            # Ensure value_mask has the same spatial dimensions as target
            if value_mask.shape[-2:] != target.shape[-2:]:
                value_mask = F.interpolate(
                    value_mask.unsqueeze(0), size=target.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0)

            # Compute the text similarity between the predictions and the labels
            value_names_z = self.encoder(names).to(self.device)
            value_names_z = value_names_z if value_names_z.dim() == 2 else value_names_z.unsqueeze(0)
            value_names_z = value_names_z.unsqueeze(1)
            similarity = F.relu(F.cosine_similarity(classes_z, value_names_z, dim=-1))

            # Extract the class indexes
            cls_idx = torch.unique(target).long()
            cls_idx = cls_idx[cls_idx != self.ignore_index]  # Remove ignore index if present

            value_mask = value_mask.argmax(dim=0)
            value_scores = similarity[:, cls_idx][value_mask.long()].permute(2, 0, 1)
            # rows, cols = value_mask.flatten(), target.flatten()
            # value_scores = similarity[rows, cols].reshape(value_mask.shape)

            for i, idx in enumerate(cls_idx):

                mask = target == idx

                recall.append(torch.sum(value_scores[i][mask]) / torch.sum(mask))
                target_idxs.append(idx)

        recall = torch.tensor(recall, device=self.device)
        target_idxs = torch.tensor(target_idxs, device=self.device)

        self.recall = torch.cat([self.recall, recall])
        self.target_idx = torch.cat([self.target_idx, target_idxs])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.average == "micro":
            return torch.mean(self.recall)
        elif self.average == "macro":
            recalls = []
            for idx in torch.unique(self.target_idx):
                mask = self.target_idx == idx
                recalls.append(torch.mean(self.recall[mask]))

            return torch.mean(torch.stack(recalls))