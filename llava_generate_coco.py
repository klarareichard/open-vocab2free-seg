from detectron2.data.datasets import load_sem_seg
import json
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaNextProcessor,  LlavaNextForConditionalGeneration

from torch.nn import DataParallel
import torch.multiprocessing as mp
import torch
# Load the processor and model

from PIL import Image
import torch.distributed as dist
import json
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from nltk.corpus import wordnet as wn
import numpy as np

import os

from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

def update_json_with_captions(batch_updates, json_file_path):
    """
    Updates the JSON file with multiple file_name and response pairs.

    Args:
    - batch_updates (list of tuples): A list containing (file_name, response) tuples.
    - json_file_path (str): The path to the JSON file to be updated.

    Returns:
    - None
    """
    # Read existing data from the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Update the entries with new file_name and response pairs
    for file_name, response in batch_updates:
        existing_data[file_name] = response

    # Write the updated data back to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

def extract_image_id(file_name):
    # Extract the filename from the full path
    base_name = os.path.basename(file_name)
    
    # Remove the file extension
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Return the image ID
    return int(name_without_ext.split("_")[-1].lstrip('0'))
    #return name_without_ext
    

def update_json_with_captions(batch_updates, json_file_path):
    """
    Updates the JSON file with multiple file_name and response pairs.

    Args:
    - batch_updates (list of tuples): A list containing (file_name, response) tuples.
    - json_file_path (str): The path to the JSON file to be updated.

    Returns:
    - None
    """
    # Try to load existing data from the JSON file
    try:
        with open(json_file_path, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or cannot be decoded, initialize as an empty list
        existing_data = []

    # Ensure existing_data is a list
    if not isinstance(existing_data, list):
        raise TypeError(f"Expected a list in the JSON file, but got {type(existing_data).__name__}")

    # Create a dictionary from existing_data for faster lookup
    existing_data_dict = {entry["file_name"]: entry for entry in existing_data if "file_name" in entry}

    for file_name, response in batch_updates:
        if file_name in existing_data_dict:
            # Update existing entry
            existing_data_dict[file_name]["caption"] = response
        else:
            # Add new entry
            existing_data_dict[file_name] = {"file_name": file_name, "caption": response}

    # Convert the dictionary back to a list
    updated_data = list(existing_data_dict.values())

    # Write the updated data back to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(updated_data, f, indent=4)
    
def save_captions_to_json(dataset_dicts, output_file):
    # Extract the required fields
    data_to_save = [{"file_name": data["file_name"], "caption": data["captions"]} for data in dataset_dicts]

    # Save to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=4)

def process_batch_on_gpu(gpu_id, model, processor, dataset_dicts, start_idx, end_idx, captions, batch_size):
    #dataset_dicts = load_sem_seg(image_dir, gt_dir, gt_ext="png", image_ext="jpg")
    
    classname_file = "/root/open_vocabulary_segmentation/datasets/coco-stuff/coco.json"
    with open(classname_file, 'r') as f_in:
        classnames = json.load(f_in)
        
    json_file = f"llava-1.6-predicted_classes_and_adjectives_ade_validation{gpu_id}.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            existing_captions = json.load(f)
    else:
        existing_captions = {}

        
    adjectives_per_class = {}
    index = 0
    # Move the model to the specific GPU
    device = torch.device(f'cuda:{gpu_id}')
    model = model.to(device)
    
    batch_images = []
    batch_prompts = []
    batch_image_ids = []
    batch_file_names = []
    #batch_gt = []

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
        
        # Check if caption already exists before processing
        if file_name in existing_captions:
            print(f"Caption for {file_name} already exists. Skipping LLAVA query...")
            continue  # Skip this image if it already has a caption
        
        text2 = "The output should be in this form: {object1: [adjective1, adjective2, ..., ], object2: [adjective3, adjective4, ..., ], ..., }."
        text3 = "This is an example how the output should look. {giraffe: [tall, brown, spotted, interacting], tree: [tall, green, leafy]}"
        conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": f"The objects in the image are: {gt_text}. Please generate a short list of adjectives for each object that describe how the object looks in the image. " + text3
                            },
                        ],
                    },
                ]
    
        
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
   #     
        batch_images.append(img)
        batch_prompts.append(prompt)
        batch_image_ids.append(image_id)
        batch_file_names.append(file_name)
        

        if len(batch_images) == batch_size or idx == end_idx - 1:
            # Process the batch
            inputs = processor(batch_prompts, batch_images, return_tensors="pt", padding=True)

            # Move inputs to the GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate the response
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=350)

            # Decode the response and store results
            responses = [processor.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Decode the response and store results
            
            #objects_list = extract_objects_from_response(response)

        # Formulate the next question based on the objects identified
            
            # Debugging: Print out responses and batch_image_ids
            #print(f"Batch responses: {responses}")
            #print(f"Batch image IDs: {batch_image_ids}")

            batch_updates = []
            for file_name, response in zip(batch_file_names, responses):
                #captions[image_id] = response
                batch_updates.append((file_name, response))
                #update_json_with_caption(file_name, response, "llava-1.6-predicted_classes_coco_train"+str(gpu_id)+".json")
            update_json_with_captions(batch_updates, f"llava-1.6-predicted_classes_coco_train{gpu_id}.json")

            
            # Clear the batch lists
            batch_images = []
            batch_prompts = []
            batch_image_ids = []
            batch_file_names = []
            #batch_responses = []

    #for idx in range(start_idx, end_idx):
        
    #    data = dataset_dicts[idx]
    #    img = Image.open(data["file_name"])
    #    file_name = data["file_name"]
    #    image_id = extract_image_id(file_name)

    #    conversation = [
    #        {
    #            "role": "user",
    #            "content": [
    #                {"type": "image"},
    #                {"type": "text", "text": "Which objects are in the image? Please describe how they look in the image."},
    #            ],
    #        },
    #    ]
    #    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process the image and prompt
    #    inputs = processor(prompt, img, return_tensors="pt")

        # Move inputs to the GPU
    #    inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate the response
    #    with torch.no_grad():
    #        outputs = model.generate(**inputs, max_new_tokens=350)

        # Decode the response
    #    response = processor.decode(outputs[0], skip_special_tokens=True)
        # Store the response
        #captions[image_id] = response
        
    #    update_json_with_caption(file_name, response, "llava-1.6-captions_val2017_coco_all"+str(gpu_id)+".json")
        
        
            

def parallel_process_images(model, processor, dataset_dicts, batch_size):
    num_gpus = torch.cuda.device_count()
    num_images = len(dataset_dicts)
    
    # Determine the number of images per GPU
    images_per_gpu = num_images // num_gpus

    processes = []

    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        end_idx = start_idx + images_per_gpu if i != num_gpus - 1 else num_images
        
        # Create a process for each GPU
        p = mp.Process(target=process_batch_on_gpu, args=(i, model, processor, dataset_dicts, start_idx, end_idx, captions, batch_size))
        p.start()
        processes.append(p)

    # Join all processes
    for p in processes:
        p.join()

    #print(captions)
    return dict(captions)




if __name__ == '__main__':

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor.tokenizer.padding_side = 'left'
    #tokenizer.padding_side = 'left'
    # LlavaForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16) #load_in_4bit=True)
    #model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))  # List of available GPUs
    #model = nn.DataParallel(model, device_ids=device_ids)
    #model.to(device)
    #generate_adjectives_llava(gt_dir, image_dir, processor, model, classfile)
    #generate_adjectives_llava(gt_dir, image_dir, processor, model, classname_file)

    mp.set_start_method('spawn', force=True)  # For compatibility with CUDA

    gt_dir = "/root/open_vocabulary_segmentation/datasets/coco-stuff/annotations_detectron2/train2017"
    image_dir = "/root/open_vocabulary_segmentation/datasets/coco-stuff/images/train2017"
    #class_json = "/root/open_vocabulary_segmentation/datasets/coco-stuff/coco.json"
    
    #class_texts = []
    #with open(class_json, 'r') as f_in:
    #    class_texts = json.load(f_in)
    # Run the parallel processing
    dataset_dicts = load_sem_seg(gt_dir, image_dir, gt_ext="png", image_ext="jpg")
    captions = mp.Manager().dict()
    captions = parallel_process_images(model, processor, dataset_dicts, 1)#dataset_dicts, batch_size)
    
    
    #for dataset_dict in dataset_dicts:
    #    file_name = dataset_dict["file_name"]
    #    image_id = extract_image_id(file_name)
    #    if image_id in captions:
    #        dataset_dict["captions"] = captions[image_id]
        #if image_id in adjectives_per_class:
        #    dataset_dict["adjectives"] = adjectives_per_class[image_id]
        
        
    #save_captions_to_json(dataset_dicts, "llava-1.6-captions_val2017_coco_all.json")
    