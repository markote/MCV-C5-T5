import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask
from pycocotools.cocoeval import COCOeval

import argparse
import sys
import glob
import os
import json

def subimage_gen(root_directory):
    pattern = os.path.join(root_directory, '**', '*.png')
    png_files = glob.glob(pattern, recursive=True)
    return sorted(png_files)

def load_images(full_array_path):
    result = []
    result_size = []
    for path_im in full_array_path:
        try: 
            with Image.open(path_im) as image:
                result_size.append(image.size[::-1])
                result.append(np.array(image))
        except (IOError, SyntaxError) as e:
            print(f"Error opening image: {e}")
            sys.exit(1)

    return full_array_path, result, result_size

def draw_predictions(image_path, predictions, output_path, id2label, gt_struct):
    frame_id = int ((image_path.split("/")[-1]).split(".")[0])
    video_id = int ((image_path.split("/")[-2])) * 10000
    image_id = frame_id + video_id
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Get the predictions
    scores = predictions['scores']
    boxes = predictions['boxes']
    labels = predictions['labels']
    
    # Iterate through each prediction and draw the bounding boxes
    for i, score in enumerate(scores):

        # Get the bounding box and class label
        box = boxes[i]
        label = labels[i].item()
        
        # Convert bounding box to integer
        x_min, y_min, x_max, y_max = box
        # Draw the bounding box (blue color)
        # Draw the bounding box (blue color)
        if id2label[label] == 'car':
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        if score > 0.5:
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        # Display the label (class) and confidence score
        # label_text = f"Class {id2label[label]} {score:.2f}"
        # cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if gt_struct is not None and image_id in gt_struct:
        for bb, label in gt_struct[image_id]:
            x_min,y_min,w,h = bb
            x_max = x_min + w
            y_max = y_min+h
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # Display the label (class) and confidence score
            # label_text = f"Class {id2label[label]}"
            # cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # Save the output image
    cv2.imwrite(output_path, image)
        
    print(f"Image saved to {output_path}")

def split_array_by_size(arr, partition_size):
    partitions = [arr[i:i + partition_size] for i in range(0, len(arr), partition_size)]
    return partitions

def process_coco_json(path_json):
    """
    Given a COCO format JSON file, returns a dictionary with image id as key and a list of pairs
    (bounding box in COCO format, label id) as value.
    
    Args:
        path_json (str): Path to the COCO format JSON file.
    
    Returns:
        dict: A dictionary with image_id as the key and a list of (bbox, label_id) pairs.
    """
    # Load JSON data
    with open(path_json, 'r') as file:
        coco_data = json.load(file)
    
    # Initialize dictionary to hold results
    image_data = {}

    # Iterate through annotations and collect bbox and label id pairs
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']  # [x_min, y_min, width, height] (COCO format)
        label_id = annotation['category_id']

        if image_id not in image_data:
            image_data[image_id] = []

        # Append bbox and label_id as a tuple
        image_data[image_id].append((bbox, label_id))

    return image_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', action='store_true', help="Enable metric computation")
    parser.add_argument('--json_output', required=False, default="./inference_finetune_DETR.json", help='Output xml')
    args = parser.parse_args()
    
    if args.metric:
        eval_gt_path = "./eval_gt.json"
        gt_struct = process_coco_json(eval_gt_path)
    else:
        gt_struct = None
    
    test_path = "/ghome/c5mcv05/team5_split_KITTI-MOTS/eval/0020" #"./test_images/" #"/ghome/c5mcv05/team5_split_KITTI-MOTS/eval/" #"/ghome/c5mcv05/datasets/KITTI-MOTS/testing/image_02/"
    output_dir= "detr_finetuned_kt_mt_best_from_trainer/"
    model = AutoModelForObjectDetection.from_pretrained(output_dir)
    processor = AutoImageProcessor.from_pretrained(output_dir)
    id2label = model.config.id2label

    array_images = subimage_gen(test_path)
    split_images = split_array_by_size(array_images, 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    json_predict = []
    j = 0
    for i, split in enumerate(split_images):
        print(f"Begin split {i}")
        path_image, images, images_size = load_images(split)
        inputs = processor(images=images, return_tensors="pt")# need to pass the annotaion in a format DETR on COCO expects, so class number between 0 and 79 for metric evaluation
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor(images_size) #[image.size[::-1]]
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes) #, threshold=0.8) increase shit

        #save_coco_json(im_result, args.json_output)
        for im_result in results:
            new_path = array_images[j]
            new_path = new_path.replace(test_path, "./pred_finetune_DETR/", 1) 
            print(f"Image: {array_images[j]}")
            
            draw_predictions(array_images[j], im_result, f"./pred_finetune_DETR/image{j:06d}.png", id2label, gt_struct)
            j += 1
        
        print(f"End split {i}")