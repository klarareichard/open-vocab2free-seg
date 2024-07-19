#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

import os
import os.path as osp
from pathlib import Path
import tqdm
from glob import glob

import numpy as np
from PIL import Image


ACDC_CATEGORIES = \
[
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    {'name': 'unlabeled'            , 'id':  0 ,      'trainId': 0 , 'color' : (  0,  0,  0) },
    {'name': 'ego vehicle'          , 'id':  1 ,      'trainId': 1 , 'color' : (  0,  0,  0) },
    {'name': 'rectification border' , 'id':  2 ,      'trainId': 2 , 'color' : (  0,  0,  0) },
    {'name': 'out of roi'           , 'id':  3 ,      'trainId': 3 , 'color' : (  0,  0,  0) },
    {'name': 'static'               , 'id':  4 ,      'trainId': 4 , 'color' : (  0,  0,  0) },
    {'name': 'dynamic'              , 'id':  5 ,      'trainId': 5 , 'color' : (111, 74,  0) },
    {'name': 'ground'               , 'id':  6 ,      'trainId': 6 , 'color' : ( 81,  0, 81) },
    {'name': 'road'                 , 'id':  7 ,      'trainId': 7 , 'color' : (128, 64,128) },
    {'name': 'sidewalk'             , 'id':  8 ,      'trainId': 8 , 'color' : (244, 35,232) },
    {'name': 'parking'              , 'id':  9 ,      'trainId': 9 , 'color' : (250,170,160) },
    {'name': 'rail track'           , 'id': 10 ,      'trainId': 10 , 'color' : (230,150,140) },
    {'name': 'building'             , 'id': 11 ,      'trainId': 11 , 'color' : ( 70, 70, 70) },
    {'name': 'wall'                 , 'id': 12 ,      'trainId': 12 , 'color' : (102,102,156) },
    {'name': 'fence'                , 'id': 13 ,      'trainId': 13 , 'color' : (190,153,153) },
    {'name': 'guard rail'           , 'id': 14 ,      'trainId': 14 , 'color' : (180,165,180) },
    {'name': 'bridge'               , 'id': 15 ,      'trainId': 15 , 'color' : (150,100,100) },
    {'name': 'tunnel'               , 'id': 16 ,      'trainId': 16 , 'color' : (150,120, 90) },
    {'name': 'pole'                 , 'id': 17 ,      'trainId': 17 , 'color' : (153,153,153) },
    {'name': 'polegroup'            , 'id': 18 ,      'trainId': 18 , 'color' : (153,153,153) },
    {'name': 'traffic light'        , 'id': 19 ,      'trainId': 19 , 'color' : (250,170, 30) },
    {'name': 'traffic sign'         , 'id': 20 ,      'trainId': 20 , 'color' : (220,220,  0) },
    {'name': 'vegetation'           , 'id': 21 ,      'trainId': 21 , 'color' : (107,142, 35) },
    {'name': 'terrain'              , 'id': 22 ,      'trainId': 22 , 'color' : (152,251,152) },
    {'name': 'sky'                  , 'id': 23 ,      'trainId': 23 , 'color' : ( 70,130,180) },
    {'name': 'person'               , 'id': 24 ,      'trainId': 24 , 'color' : (220, 20, 60) },
    {'name': 'rider'                , 'id': 25 ,      'trainId': 25 , 'color' : (255,  0,  0) },
    {'name': 'car'                  , 'id': 26 ,      'trainId': 26 , 'color' : (  0,  0,142) },
    {'name': 'truck'                , 'id': 27 ,      'trainId': 27 , 'color' : (  0,  0, 70) },
    {'name': 'bus'                  , 'id': 28 ,      'trainId': 28 , 'color' : (  0, 60,100) },
    {'name': 'caravan'              , 'id': 29 ,      'trainId': 29 , 'color' : (  0,  0, 90) },
    {'name': 'trailer'              , 'id': 30 ,      'trainId': 30 , 'color' : (  0,  0,110) },
    {'name': 'train'                , 'id': 31 ,      'trainId': 31 , 'color' : (  0, 80,100) },
    {'name': 'motorcycle'           , 'id': 32 ,      'trainId': 32 , 'color' : (  0,  0,230) },
    {'name': 'bicycle'              , 'id': 33 ,      'trainId': 33 , 'color' : (119, 11, 32) },
    {'name': 'license plate'        , 'id': -1 ,      'trainId': -1 , 'color' : (  0,  0,142) }]

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "acdc"
    print("dataset_dir")

    id_map = {}
    for cat in ACDC_CATEGORIES:
        id_map[cat["id"]] = cat["trainId"]

    for name in ["train", "val"]:
        annotation_dir = dataset_dir / "annotations" / name
        output_dir = dataset_dir / "annotations_detectron2" / name / "random"
        print(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            if not ((str(file).split('/')[-1])[0] == '.'):
                output_file = output_dir / file.name
                print(str(file))
                lab = np.asarray(Image.open(file))
                assert lab.dtype == np.uint8

                output = np.zeros_like(lab, dtype=np.uint8)
                for obj_id in np.unique(lab):
                    if obj_id in id_map:
                        output[lab == obj_id] = id_map[obj_id]

                if "datasets/acdc/annotations/val/GP010607_frame_000968_gt_labelTrainIds.png" == str(file):
                    print(output)
                Image.fromarray(output).save(output_file)