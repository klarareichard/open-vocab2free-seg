# Adapted from https://github.com/altndrr/vicss/

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassRecall
from typing import List, Tuple

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
            target = torch.tensor(target, device=self.device)

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
            target = torch.tensor(target, device=self.device)

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
            target = torch.tensor(target, device=self.device)
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