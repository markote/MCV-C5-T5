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

def compute_metric(gt_json, pred_json):
    print("Final metric result: ")
    COCO_gt = COCO(gt_json)
    valid_image_ids = set(COCO_gt.getImgIds())

    print(f"Size: {len(pred_json)}")
    filtered_predictions = [
        pred for pred in pred_json if pred['image_id'] in valid_image_ids
    ]
    print(f"Size: {len(filtered_predictions)}")
    coco_pred_filtered = COCO_gt.loadRes(filtered_predictions)

    coco_eval = COCOeval(COCO_gt, coco_pred_filtered, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def increase_json(json_predict, path_im, predictions):
    scores = predictions['scores']
    boxes = predictions['boxes']
    labels = predictions['labels']
    frame_id = int ((path_im.split("/")[-1]).split(".")[0])
    video_id = int ((path_im.split("/")[-2])) * 10000
    image_id = frame_id + video_id
    for i, score in enumerate(scores):
        label = labels[i].item()
        box = boxes[i]
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        bbox = [x_min.item(), y_min.item(), w.item(), h.item()]
        # print(f"boxes x,y,w,h {type(bbox[0]),type(bbox[1]),type(bbox[2]),type(bbox[3])}")
        # print(f"score: {type(score)}")
        json_predict.append({ "image_id":image_id, "category_id": label, "bbox": bbox, "score": score.item() })

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

def split_array_by_size(arr, partition_size):
    partitions = [arr[i:i + partition_size] for i in range(0, len(arr), partition_size)]
    return partitions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--json_output', required=False, default="./inference_finetune_DETR.json", help='Output xml')
    args = parser.parse_args()
    
    eval_gt_path = "./eval_gt.json"
    eval_path = "/ghome/c5mcv05/team5_split_KITTI-MOTS/eval/"
    output_dir= "detr_finetuned_kt_mt_best_from_trainer/"
    model = AutoModelForObjectDetection.from_pretrained(output_dir)
    processor = AutoImageProcessor.from_pretrained(output_dir)
    
    array_images = subimage_gen(eval_path)
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
        #print(f"TArget size: {target_sizes}")
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes) #, threshold=0.8) increase shit

        #save_coco_json(im_result, args.json_output)
        for im_result in results:
            increase_json(json_predict, array_images[j], im_result)
            j += 1
        
        print(f"End split {i}")
    
    with open(args.json_output, 'w') as json_file:
        json.dump(json_predict, json_file, indent=4)
    compute_metric(eval_gt_path, json_predict)
