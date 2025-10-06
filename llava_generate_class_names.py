import time
import re
import json
import os
from collections import Counter
from PIL import Image

import torch
import torch.multiprocessing as mp
import numpy as np
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from detectron2.data.datasets import load_sem_seg
import argparse


# ---------------------
# Utility functions
# ---------------------

def extract_image_id(file_name):
    base_name = os.path.basename(file_name)
    return os.path.splitext(base_name)[0]


def update_json_with_class_names(file_name, class_names, json_file):
    """Add or update the entry for an image with its detected class names."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    entry_found = False
    for entry in data:
        if entry["file_name"] == file_name:
            entry["class_names"] = class_names
            entry_found = True
            break

    if not entry_found:
        data.append({"file_name": file_name, "class_names": class_names})

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


# ---------------------
# Parsing
# ---------------------

def parse_objects_list(caption):
    """
    Extract a list of object names from the model's response.
    Expected format after 'ASSISTANT:': [cat, dog, sheep]
    """
    parts = caption.split("ASSISTANT:")
    if len(parts) != 2:
        return None

    pattern = r'\[(.*?)\]'
    match = re.search(pattern, parts[1])
    if not match:
        return None

    list_string = match.group(1)
    parsed_list = [word.strip().lower().replace('*', '') for word in list_string.split(',')]
    return parsed_list


def is_failure_case_objects(parsed_list):
    """Define conditions under which the parsed list is considered invalid."""
    if not parsed_list:
        return True
    counts = Counter(parsed_list)
    if any(c > 1 for c in counts.values()):  # duplicated objects
        return True
    if len(parsed_list) > 50:
        return True
    return False


# ---------------------
# Query
# ---------------------

def retry_query_until_success_objects(processor, model, conversation, img, device, max_tokens=350):
    """Repeatedly query the model until a valid object list is returned."""
    while True:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(response)
        objects_list = parse_objects_list(response)
        if objects_list and not is_failure_case_objects(objects_list):
            return objects_list
        time.sleep(2)


# ---------------------
# Main processing logic
# ---------------------

def process_batch_on_gpu(gpu_id, model, processor, dataset_dicts, start_idx, end_idx, batch_size, dataset_short_name):
    dataset_root = os.environ.get("DETECTRON2_DATASETS", "datasets")
    classname_file = os.path.join(dataset_root, dataset_short_name + ".json")
    with open(classname_file, 'r') as f_in:
        classnames = json.load(f_in)

    json_file = dataset_short_name + f"_c_llava{gpu_id}.json"

    device = torch.device(f'cuda:{gpu_id}')
    model = model.to(device)

    for idx in range(start_idx, end_idx):
        data = dataset_dicts[idx]
        img = Image.open(data["file_name"])
        file_name = data["file_name"]

        # Build conversation prompt for object detection
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Which objects are in the image? "
                            "If there is more than one object of the same kind, "
                            "only report the object once. "
                            "The list should look like this example: [cat, dog, sheep, car, motorcycle]."
                        ),
                    },
                ],
            },
        ]

        objects_list = retry_query_until_success_objects(processor, model, conversation, img, device)
        update_json_with_class_names(file_name, objects_list, json_file)


def parallel_process_images(model, processor, dataset_dicts, batch_size, dataset_short_name):
    num_gpus = torch.cuda.device_count()
    num_images = len(dataset_dicts)
    images_per_gpu = num_images // num_gpus
    processes = []

    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        end_idx = start_idx + images_per_gpu if i != num_gpus - 1 else num_images
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(i, model, processor, dataset_dicts, start_idx, end_idx, batch_size, dataset_short_name)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def merge_gpu_json_files(dataset_short_name, num_gpus):
    merged = {}
    for gpu_id in range(num_gpus):
        file_path = dataset_short_name + f"_c_gt_llava{gpu_id}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    merged[entry["file_name"]] = entry["class_names"]

    # Save merged JSON
    merged_file = dataset_short_name + "_c_gt_llava.json"
    merged_list = [{"file_name": k, "class_names": v} for k, v in merged.items()]
    with open(merged_file, 'w') as f:
        json.dump(merged_list, f, indent=4)
    print(f"Merged JSON saved to {merged_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_short_name", type=str, required=True, help="Short name of the dataset")
    args = parser.parse_args()

    dataset_short_name = args.dataset_short_name
    mp.set_start_method('spawn', force=True)

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16
    )

    processor.tokenizer.padding_side = 'left'

    dataset_root = os.environ.get("DETECTRON2_DATASETS", "datasets")
    dataset_subdir_ground_truth = os.environ.get("DATASET_SUBDIR_GROUNDTRUTH", "subdir_groundtruth")
    dataset_subdir_images = os.environ.get("DATASET_SUBDIR_IMAGES", "subdir_images")
    #dataset_short_name = "ade150"

    dataset_dicts = load_sem_seg(
        os.path.join(dataset_root, dataset_subdir_ground_truth),
        os.path.join(dataset_root, dataset_subdir_images),
        gt_ext="png",
        image_ext="jpg"
    )

    parallel_process_images(model, processor, dataset_dicts, batch_size=1, dataset_short_name=dataset_short_name)

    # Merge per-GPU JSON files into a single final JSON
    merge_gpu_json_files(dataset_short_name, num_gpus=torch.cuda.device_count())
