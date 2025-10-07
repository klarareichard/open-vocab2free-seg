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

from .metrics import SemanticJaccardIndex, SemanticRecall, SemanticWeightedJaccardIndex

# Usage example within a process function

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
        self.hji = SemanticJaccardIndex(mode="hard", classes=self._class_names, ignore_index=self._ignore_label).to(
            self._device)
        #self.sji = SemanticJaccardIndex(mode="soft", classes=self._class_names, ignore_index=self._ignore_label).to(
        #    self._device)
        self.hr = SemanticRecall(mode="hard", classes=self._class_names, ignore_index=self._ignore_label).to(
            self._device)
        self.sr = SemanticRecall(mode="soft", classes=self._class_names, ignore_index=self._ignore_label).to(
            self._device)

        # New HJI for mapped classes
        self.mapped_hji = SemanticJaccardIndex(mode="hard", classes=self._class_names,
                                               ignore_index=self._ignore_label).to(self._device)


    def reset(self):
        self.hji.reset()
        #self.sji.reset()
        self.hr.reset()
        self.sr.reset()
        self.mapped_hji.reset()


    def process(self, inputs, outputs):
        # Define the path to your JSON file
        for input, output in zip(inputs, outputs):
            pred_classes = input["class_names"]
            # pred_classes = [s.strip("'") for s in pred_classes.strip("[]").split(", ")]
            #pred_classes = [*{*pred_classes}]
            pred_mask = output["sem_seg"].cpu().numpy()
            pred = [(pred_classes, pred_mask)]

            mapped_classes = input.get("mapped_class_names", "")
            if mapped_classes:
                map_pred = [(mapped_classes, pred_mask)]

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.load_gt_sem_seg(gt_filename)
            gt = [gt]

            # Update metrics
            self.hji.update(pred, gt)
            self.hr.update(pred, gt)
            self.sr.update(pred, gt)

            # Update SJI based on the presence of mapped classes
            if mapped_classes:
                self.mapped_hji.update(map_pred, gt)
            #else:
            #    self.sji.update(pred, gt)

    def evaluate(self):
        if self._distributed:
            synchronize()
            metric_states = all_gather({
                'hji': self.hji.state_dict(),
                #'sji': self.sji.state_dict(),
                'hr': self.hr.state_dict(),
                'sr': self.sr.state_dict(),
                'mapped_hji': self.mapped_hji.state_dict()
            })

            if not is_main_process():
                return

            # Combine metric states
            for metric_name in ['hji', 'hr', 'sr', 'mapped_hji']:
                getattr(self, metric_name).load_state_dict(
                    self._combine_states([state[metric_name] for state in metric_states])
                )

        hji = self.hji.compute()
        #sji = self.sji.compute()
        hr = self.hr.compute()
        sr = self.sr.compute()
        mapped_hji = self.mapped_hji.compute()
        weighted_sji = self.weighted_jaccard_index.compute()

        res = OrderedDict()
        res["sem_seg"] = {
            "HJI": hji.item() * 100,
            "SJI": mapped_hji.item() * 100, #if mapped_hji.item() != 0 else sji.item() * 100,
            "HR": hr.item() * 100,
            "SR": sr.item() * 100
        }

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.json")
            with PathManager.open(file_path, "w") as f:
                json.dump(res, f)

        results = OrderedDict({"sem_seg": res["sem_seg"]})
        self._logger.info(results)
        
        
        if self._dataset_name:
            output_path = os.path.join(self._output_dir, self._dataset_name)
            PathManager.mkdirs(output_path)
            output_file = os.path.join(output_path, "results.json")
        

        # Write the OrderedDict to the JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        
        print(results)
        return results

    @staticmethod
    def _combine_states(states):
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
        return np.array(Image.open(gt_filename))