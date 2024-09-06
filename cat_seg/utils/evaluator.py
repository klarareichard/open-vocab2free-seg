import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from collections import OrderedDict
import os
import json
import logging
from PIL import Image

from .metrics import SemanticJaccardIndex, SemanticRecall


class VocabFreeEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics for Vocabulary Free pipeline.
    """

    def __init__(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
            *,
            sem_seg_loading_fn=None,
            num_classes=None,
            ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
            num_classes, ignore_label: deprecated arguments
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        self._class_names = meta.stuff_classes
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # Initialize metrics on the correct device
        self.jaccard_index = SemanticJaccardIndex(mode="soft", classes=self._class_names).to(self._device)
        self.recall = SemanticRecall(mode="soft", classes=self._class_names).to(self._device)

    def reset(self):
        self.jaccard_index.reset()
        self.recall.reset()

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is a list of dicts with key
                "sem_seg" that contains semantic segmentation prediction.
        """
        for input, output in zip(inputs, outputs):
            pred_classes =  input["class_names"] # [s.strip("'") for s in input["class_names"].strip("[]").split(", ")]
            pred_classes = [*{*pred_classes}]
            pred_mask = output["sem_seg"].cpu().numpy()#.to(self._device).
            pred = [(pred_classes, pred_mask)]  # Wrap in list for batch of size 1

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.load_gt_sem_seg(gt_filename)#torch.from_numpy(np.load(gt_filename)).to(self._device)  # Load and move to device
            gt = [gt]  # Wrap in list for batch of size 1

            # Update metrics
            self.jaccard_index.update(pred, gt)
            self.recall.update(pred, gt)

    def evaluate(self):
        if self._distributed:
            synchronize()
            # Gather metric states from all processes
            jaccard_index_state = all_gather(self.jaccard_index.state_dict())
            recall_state = all_gather(self.recall.state_dict())

            if not is_main_process():
                return

            # Combine metric states
            self.jaccard_index.load_state_dict(self._combine_states(jaccard_index_state))
            self.recall.load_state_dict(self._combine_states(recall_state))

        jaccard_index = self.jaccard_index.compute()
        recall = self.recall.compute()

        res = OrderedDict()
        res["sem_seg"] = {
            "mIoU": jaccard_index.item() * 100,
            "Recall": recall.item() * 100,
        }

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.json")
            with PathManager.open(file_path, "w") as f:
                json.dump(res, f)

        results = OrderedDict({"sem_seg": res["sem_seg"]})
        self._logger.info(results)
        return results

    @staticmethod
    def _combine_states(states):
        """
        Combine metric states from multiple processes.
        This method should be implemented based on how your metric states are structured.
        """
        combined = states[0].copy()
        for state in states[1:]:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    combined[k] += v
                elif isinstance(v, (int, float)):
                    combined[k] += v
        return combined



    @staticmethod
    def load_gt_sem_seg(gt_filename: str) -> np.ndarray:
        """
        Load ground truth semantic segmentation mask.

        Args:
            gt_filename (str): Path to the ground truth semantic segmentation mask.

        Returns:
            np.ndarray: Ground truth semantic segmentation mask.
        """
        return np.array(Image.open(gt_filename))
