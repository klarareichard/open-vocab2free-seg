import argparse
import time
import re
import json
import os
from collections import defaultdict, Counter
from PIL import Image

import nltk
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import torch.multiprocessing as mp
from detectron2.data.datasets import load_sem_seg
import numpy as np


def merge_json_files(dataset_short_name, num_gpus):
    merged_data = []
    for gpu_id in range(num_gpus):
        json_file = f"{dataset_short_name}_c_gt_a_llava{gpu_id}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                merged_data.extend(data)
            os.remove(json_file)  # optional: remove per-GPU file after merging

    final_json_file = f"{dataset_short_name}_c_gt_a_llava.json"
    with open(final_json_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Merged JSON saved to {final_json_file}")

def extract_image_id(file_name):
    # Extract the base filename without extension
    base_name = os.path.basename(file_name)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext


def update_json_with_class_names(file_name, class_names, json_file):
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


def update_json_with_attributes_list(file_name, attributes_list, json_file):
    print(json_file)
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    entry_found = False
    for entry in data:
        if entry["file_name"] == file_name:
            entry["attributes_list"] = attributes_list
            entry_found = True
            break

    if not entry_found:
        data.append({"file_name": file_name, "attributes_list": attributes_list})

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def parse_to_dict(input_string):
    input_string = input_string.replace("\n", ", ")
    parts = input_string.split("ASSISTANT:")
    if len(parts) != 2:
        return None, None

    input_string = parts[1]
    result_dict = defaultdict(list)
    pattern = r'([^:]+):\s*\[(.*?)\]'
    matches = re.findall(pattern, input_string)

    if not matches:
        return None

    for key, value in matches:
        key = key.strip(" ,{}").strip().lower().replace('*', '')
        values = [v.strip().lower().replace('*', '') for v in value.split(',')]
        result_dict[key].extend(values)

    return dict(result_dict)


def remove_none_not_visible(attributes_dict):
    new_dict = {}
    invalid_values = set([
        "none", "not visible", "none visible", "invisible", "no visible",
        "empty", "not detected", "no detected", "no", "unidentified",
        "not applicable", "nonexistent"
    ])
    for key, vals in attributes_dict.items():
        for val in vals:
            if val.lower() not in invalid_values:
                if key not in new_dict:
                    new_dict[key] = []
                new_dict[key].append(val)
    return new_dict


def is_failure_case_adjectives(attributes_dict):
    if not attributes_dict:
        return True

    attributes_dict = remove_none_not_visible(attributes_dict)
    if not attributes_dict:
        return True

    for obj, adjectives in attributes_dict.items():
        if len(adjectives) > 50:
            return True
        counts = Counter(adjectives)
        if any(c > 1 for c in counts.values()):
            return True

    return False


def is_failure_case(parsed_list_object, parsed_list_adjective):
    return all(cls not in parsed_list_object for cls in parsed_list_adjective.keys())


def is_adjective(word):
    pos_tag = nltk.pos_tag([word])[0][1]
    return pos_tag in ['JJ', 'JJR', 'JJS']


def is_failure_case_objects(parsed_list):
    if not parsed_list:
        return True
    counts = Counter(parsed_list)
    if any(c > 1 for c in counts.values()):
        return True
    if len(parsed_list) > 50:
        return True
    for cls in parsed_list:
        if cls.split() and is_adjective(cls.split()[0]):
            return True
    return False


def parse_objects_list(caption):
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


def retry_query_until_success_adjectives(processor, model, conversation, img, device, max_tokens=350):
    while True:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(response)
        adjectives_list = parse_to_dict(response)
        print(adjectives_list)
        if adjectives_list and not is_failure_case_adjectives(adjectives_list):
            return adjectives_list
        time.sleep(2)


def retry_query_until_success_objects(processor, model, conversation, img, device, max_tokens=350):
    while True:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(prompt, img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        objects_list = parse_objects_list(response)
        if objects_list and not is_failure_case_objects(objects_list):
            return objects_list
        time.sleep(2)


def load_existing_captions(base_file):
    if os.path.exists(base_file):
        try:
            with open(base_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def process_batch_on_gpu(gpu_id, model, processor, dataset_dicts, start_idx, end_idx, batch_size, dataset_short_name):
    dataset_root = os.environ.get("DETECTRON2_DATASETS", "datasets")
    classname_file = os.path.join(dataset_root, dataset_short_name+".json")
    with open(classname_file, 'r') as f_in:
        classnames = json.load(f_in)
    #base_file = os.path.join(dataset_root, "a-847_c_gt.json")#"CAT-Seg/a-847_c_gt.json")
    #existing_captions = load_existing_captions(base_file)
    json_file = dataset_short_name+f"_c_gt_a_llava{gpu_id}.json"

    device = torch.device(f'cuda:{gpu_id}')
    model = model.to(device)

    for idx in range(start_idx, end_idx):
        data = dataset_dicts[idx]
        img = Image.open(data["file_name"])
        file_name = data["file_name"]
        image_id = extract_image_id(file_name)
        sem_seg_filename = data.get("sem_seg_file_name", "")
        sem_seg_image = Image.open(sem_seg_filename)
        
        # Convert the image to a NumPy array
        sem_seg_array = np.array(sem_seg_image)
        unique_classes = np.unique(sem_seg_array)
        #if gt_cls is not None:
        #gt_cls = torch.unique(torch.cat(sem_seg_array, dim=0))
        unique_classes = unique_classes[unique_classes != 255] #.to(self.device) # @TODO
        gt_text = [classnames[c] for c in unique_classes]

        #existing_entry = next((entry for entry in existing_captions if extract_image_id(entry["file_name"]) == file_name), None)
        #if existing_entry and not is_failure_case_adjectives(existing_entry.get("attributes_list", {})):
        #    continue
        text2 = "The output should be in this form: {object1: [adjective1, adjective2, ..., ], object2: [adjective3, adjective4, ..., ], ..., }."
        text3 = "This is an example how the output should look. {giraffe: [tall, brown, spotted, interacting], tree: [tall, green, leafy]}"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"The objects in the image are: {gt_text}. Please generate a short list of adjectives for each object that describe how the object looks in the image. " + text3},
                ],
            },
        ]

        #conversation = [
        #    {
        #        "role": "user",
        #        "content": [
        #            {"type": "image"},
        #            {"type": "text",
        #             "text": f"The objects in the image are: {', '.join(ram_predicted_class_names)}. "
        #                     f"Generate a short list of adjectives for each object."},
        #        ],
        #    },
        #]

        adjectives_list = retry_query_until_success_adjectives(processor, model, conversation, img, device)
        if not is_failure_case(gt_text, adjectives_list):
            update_json_with_class_names(file_name, gt_text, json_file)
            update_json_with_attributes_list(file_name, adjectives_list, json_file)


def parallel_process_images(model, processor, dataset_dicts, batch_size, dataset_short_name):
    num_gpus = torch.cuda.device_count()
    num_images = len(dataset_dicts)
    images_per_gpu = num_images // num_gpus
    processes = []

    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        end_idx = start_idx + images_per_gpu if i != num_gpus - 1 else num_images
        p = mp.Process(target=process_batch_on_gpu,
                       args=(i, model, processor, dataset_dicts, start_idx, end_idx, batch_size, dataset_short_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_short_name", type=str, required=True, help="Short name of the dataset")
    args = parser.parse_args()

    dataset_short_name = args.dataset_short_name

    mp.set_start_method('spawn', force=True)  # For compatibility with CUDA

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16
    )
    processor.tokenizer.padding_side = 'left'

    dataset_root = os.environ.get("DETECTRON2_DATASETS", "datasets")
    dataset_subdir_ground_truth = os.environ.get("DATASET_SUBDIR_GROUNDTRUTH", "subdir_groundtruth")
    dataset_subdir_images = os.environ.get("DATASET_SUBDIR_IMAGES", "subdir_images")

    dataset_dicts = load_sem_seg(
        os.path.join(dataset_root, dataset_subdir_ground_truth),
        os.path.join(dataset_root, dataset_subdir_images),
        gt_ext="png",
        image_ext="jpg"
    )

    parallel_process_images(model, processor, dataset_dicts, 1, dataset_short_name)

    # Merge per-GPU JSON files into one
    num_gpus = torch.cuda.device_count()
    merge_json_files(dataset_short_name, num_gpus)
