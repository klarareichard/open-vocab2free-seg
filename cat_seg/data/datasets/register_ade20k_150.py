import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy
import json

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"id": 92, "name": "banner", "supercategory": "textile"},
    {"id": 93, "name": "blanket", "supercategory": "textile"},
    {"id": 94, "name": "branch", "supercategory": "plant"},
    {"id": 95, "name": "bridge", "supercategory": "building"},
    {"id": 96, "name": "building-other", "supercategory": "building"},
    {"id": 97, "name": "bush", "supercategory": "plant"},
    {"id": 98, "name": "cabinet", "supercategory": "furniture-stuff"},
    {"id": 99, "name": "cage", "supercategory": "structural"},
    {"id": 100, "name": "cardboard", "supercategory": "raw-material"},
    {"id": 101, "name": "carpet", "supercategory": "floor"},
    {"id": 102, "name": "ceiling-other", "supercategory": "ceiling"},
    {"id": 103, "name": "ceiling-tile", "supercategory": "ceiling"},
    {"id": 104, "name": "cloth", "supercategory": "textile"},
    {"id": 105, "name": "clothes", "supercategory": "textile"},
    {"id": 106, "name": "clouds", "supercategory": "sky"},
    {"id": 107, "name": "counter", "supercategory": "furniture-stuff"},
    {"id": 108, "name": "cupboard", "supercategory": "furniture-stuff"},
    {"id": 109, "name": "curtain", "supercategory": "textile"},
    {"id": 110, "name": "desk-stuff", "supercategory": "furniture-stuff"},
    {"id": 111, "name": "dirt", "supercategory": "ground"},
    {"id": 112, "name": "door-stuff", "supercategory": "furniture-stuff"},
    {"id": 113, "name": "fence", "supercategory": "structural"},
    {"id": 114, "name": "floor-marble", "supercategory": "floor"},
    {"id": 115, "name": "floor-other", "supercategory": "floor"},
    {"id": 116, "name": "floor-stone", "supercategory": "floor"},
    {"id": 117, "name": "floor-tile", "supercategory": "floor"},
    {"id": 118, "name": "floor-wood", "supercategory": "floor"},
    {"id": 119, "name": "flower", "supercategory": "plant"},
    {"id": 120, "name": "fog", "supercategory": "water"},
    {"id": 121, "name": "food-other", "supercategory": "food-stuff"},
    {"id": 122, "name": "fruit", "supercategory": "food-stuff"},
    {"id": 123, "name": "furniture-other", "supercategory": "furniture-stuff"},
    {"id": 124, "name": "grass", "supercategory": "plant"},
    {"id": 125, "name": "gravel", "supercategory": "ground"},
    {"id": 126, "name": "ground-other", "supercategory": "ground"},
    {"id": 127, "name": "hill", "supercategory": "solid"},
    {"id": 128, "name": "house", "supercategory": "building"},
    {"id": 129, "name": "leaves", "supercategory": "plant"},
    {"id": 130, "name": "light", "supercategory": "furniture-stuff"},
    {"id": 131, "name": "mat", "supercategory": "textile"},
    {"id": 132, "name": "metal", "supercategory": "raw-material"},
    {"id": 133, "name": "mirror-stuff", "supercategory": "furniture-stuff"},
    {"id": 134, "name": "moss", "supercategory": "plant"},
    {"id": 135, "name": "mountain", "supercategory": "solid"},
    {"id": 136, "name": "mud", "supercategory": "ground"},
    {"id": 137, "name": "napkin", "supercategory": "textile"},
    {"id": 138, "name": "net", "supercategory": "structural"},
    {"id": 139, "name": "paper", "supercategory": "raw-material"},
    {"id": 140, "name": "pavement", "supercategory": "ground"},
    {"id": 141, "name": "pillow", "supercategory": "textile"},
    {"id": 142, "name": "plant-other", "supercategory": "plant"},
    {"id": 143, "name": "plastic", "supercategory": "raw-material"},
    {"id": 144, "name": "platform", "supercategory": "ground"},
    {"id": 145, "name": "playingfield", "supercategory": "ground"},
    {"id": 146, "name": "railing", "supercategory": "structural"},
    {"id": 147, "name": "railroad", "supercategory": "ground"},
    {"id": 148, "name": "river", "supercategory": "water"},
    {"id": 149, "name": "road", "supercategory": "ground"},
    {"id": 150, "name": "rock", "supercategory": "solid"},
    {"id": 151, "name": "roof", "supercategory": "building"},
    {"id": 152, "name": "rug", "supercategory": "textile"},
    {"id": 153, "name": "salad", "supercategory": "food-stuff"},
    {"id": 154, "name": "sand", "supercategory": "ground"},
    {"id": 155, "name": "sea", "supercategory": "water"},
    {"id": 156, "name": "shelf", "supercategory": "furniture-stuff"},
    {"id": 157, "name": "sky-other", "supercategory": "sky"},
    {"id": 158, "name": "skyscraper", "supercategory": "building"},
    {"id": 159, "name": "snow", "supercategory": "ground"},
    {"id": 160, "name": "solid-other", "supercategory": "solid"},
    {"id": 161, "name": "stairs", "supercategory": "furniture-stuff"},
    {"id": 162, "name": "stone", "supercategory": "solid"},
    {"id": 163, "name": "straw", "supercategory": "plant"},
    {"id": 164, "name": "structural-other", "supercategory": "structural"},
    {"id": 165, "name": "table", "supercategory": "furniture-stuff"},
    {"id": 166, "name": "tent", "supercategory": "building"},
    {"id": 167, "name": "textile-other", "supercategory": "textile"},
    {"id": 168, "name": "towel", "supercategory": "textile"},
    {"id": 169, "name": "tree", "supercategory": "plant"},
    {"id": 170, "name": "vegetable", "supercategory": "food-stuff"},
    {"id": 171, "name": "wall-brick", "supercategory": "wall"},
    {"id": 172, "name": "wall-concrete", "supercategory": "wall"},
    {"id": 173, "name": "wall-other", "supercategory": "wall"},
    {"id": 174, "name": "wall-panel", "supercategory": "wall"},
    {"id": 175, "name": "wall-stone", "supercategory": "wall"},
    {"id": 176, "name": "wall-tile", "supercategory": "wall"},
    {"id": 177, "name": "wall-wood", "supercategory": "wall"},
    {"id": 178, "name": "water-other", "supercategory": "water"},
    {"id": 179, "name": "waterdrops", "supercategory": "water"},
    {"id": 180, "name": "window-blind", "supercategory": "window"},
    {"id": 181, "name": "window-other", "supercategory": "window"},
    {"id": 182, "name": "wood", "supercategory": "solid"},
]


def _get_ade20k_150_meta():
    ade20k_150_classes = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass",
                          "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain",
                          "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug",
                          "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
                          "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
                          "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
                          "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
                          "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
                          "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
                          "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning",
                          "streetlight", "booth", "television receiver", "airplane", "dirt track", "apparel", "pole",
                          "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van",
                          "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
                          "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
                          "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
                          "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
                          "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower",
                          "radiator", "glass", "clock", "flag"]

    ret = {
        "stuff_classes": ade20k_150_classes,
    }
    return ret


def extract_image_id(file_name):
    # Extract the filename from the full path
    base_name = os.path.basename(file_name)

    # Remove the file extension
    name_without_ext = os.path.splitext(base_name)[0]
    name_without_ = name_without_ext.split("_")[-1]
    # Return the image ID
    return int(name_without_.lstrip('0'))


def load_ade_with_attributes(image_dir, gt_dir, adjectives_file):
    dataset_dicts = load_sem_seg(image_dir, gt_dir, gt_ext="png", image_ext="jpg")
    with open(adjectives_file, 'r') as f:
        adjectives_data = json.load(f)

    adjectives = {}
    class_names = {}
    mapped_class_names = {}
    clusters = {}
    for data in adjectives_data:
        file_name = data["file_name"]
        image_id = extract_image_id(file_name)
        data["image_id"] = image_id

        if "attributes_list" in data:
            adjectives_dict = data['attributes_list']
        else:
            adjectives_dict = None

        class_name = data['class_names']

        if "mapped_class_names" in data:
            mapped_class_name = data['mapped_class_names']
        else:
            mapped_class_name = None

        if "clusters" in data:
            cluster = data['clusters']
        else:
            cluster = None

        adjectives[image_id] = adjectives_dict
        class_names[image_id] = class_name
        mapped_class_names[image_id] = mapped_class_name
        clusters[image_id] = cluster

    for dataset_dict in dataset_dicts:
        file_name = dataset_dict["file_name"]
        image_id = extract_image_id(file_name)
        dataset_dict["image_id"] = image_id

        if image_id in adjectives:
            dataset_dict["attributes_list"] = adjectives[image_id]

        if image_id in class_names:
            dataset_dict["class_names"] = class_names[image_id]
            dataset_dict["mapped_class_names"] = mapped_class_names[image_id]
        if image_id in clusters:
            dataset_dict["clusters"] = clusters[image_id]

    return dataset_dicts


def register_ade20k_150(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_150_meta()
    extra_classes = []
    for category in meta["stuff_classes"]:
        seen = False
        for entry in COCO_CATEGORIES:
            if category == entry["name"]:
                seen = True
        if not seen:
            extra_classes.append(category)
    meta.update({"val_extra_classes": extra_classes})
    # VOCAB-FREE: ram_predicted_class_names_ade_all_tags.json #
    # OPEN-VOCAB: ram_plus_class_names_ade_only_ade_tags.json
    # VOCAB-FREE + adj: ram_predicted_class_names_ade_val_vocab_free.json
    # GT + adj: llava-1.6-gt_classes_predicted_adj_ade_val.json
    # Llava + adj: llava-1.6-predicted_classes_ade_validation_fixed.json #llava-1.6-predicted_classes_adjectives_ade_validation_fixed.json
    predicted_class_path = "c_tag_ade-150_thres_0.json" #"a-150_c_ram_a_llava_m_ST_thres_0.json" #"llava-1.6-gt_classes_predicted_adj_ade_val.json" #"a-150_c_ram_a_llava_m_ST.json" #"a-150_c_ram_a_llava_m_ST_cl_3e-1.json"#"filtered_output.json"# #"a-150_c_cased_m_ST_thres07.json"#"a-150_c_ram_a_llava_m_ST.json"# "filtered_output.json" #"ram_predicted_class_names_ade_val_vocab_free.json"#"llava-1.6-predicted_classes_ade_validation_right_fixed.json"#"llava-1.6-gt_classes_predicted_adj_ade_val.json" #"ram_plus_class_names_ade_only_ade_tags.json" # "ram_predicted_class_names_ade_all_tags.json"
    for name, image_dirname, sem_seg_dirname, attributes_list_file in [
        ("test", "images/validation", "annotations_detectron2/validation",
            predicted_class_path),
            #"ram_predicted_class_names_ade_all_tags.json"),
            #"llava-1.6-predicted_classes_ade_validation_remapped_with_llava_embed.json"),
         #"llava-1.6-gt_classes_predicted_adj_ade_val.json"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        attributes_list = os.path.join(root, attributes_list_file)
        name = f"ade20k_150_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_ade_with_attributes(y, x, attributes_list)
            # gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg",
                                      ignore_label=255, **meta, )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150(_root)
