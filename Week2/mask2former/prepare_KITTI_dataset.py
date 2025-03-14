import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
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

def increase_json(json_predict, path_im, predictions, id2id_map, remove=True):
    scores = predictions['scores']
    boxes = predictions['boxes']
    labels = predictions['labels']
    frame_id = int ((path_im.split("/")[-1]).split(".")[0])
    video_id = int ((path_im.split("/")[-2])) * 10000
    image_id = frame_id + video_id
    for i, score in enumerate(scores):
        label = id2id_map[labels[i].item()]
        if not remove or label != 252:
            box = boxes[i]
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            bbox = [x_min.item(), y_min.item(), w.item(), h.item()]
            # print(f"boxes x,y,w,h {type(bbox[0]),type(bbox[1]),type(bbox[2]),type(bbox[3])}")
            # print(f"score: {type(score)}")
            json_predict.append({ "image_id":image_id, "category_id": label, "bbox": bbox, "score": score.item() })

def split_array_by_size(arr, partition_size):
    partitions = [arr[i:i + partition_size] for i in range(0, len(arr), partition_size)]
    return partitions

def image_str_gen(str_int):
    return f"{str_int:06d}.png"

def load_gt(gt_path, output_json_path="./gt.json", path_images="./eval/"):
    pattern = os.path.join(gt_path, '*.txt')
    txt_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Json gt: {txt_files}")
    json_dict = {}
    json_dict["images"] = []
    json_dict["annotations"] = []
    json_dict["categories"] = [{ "id": 0, "name": "car" }, { "id": 1, "name": "pedestrian" }]
    seen_images = set()
    counter = 0
    count_detections = dict()
    for txt_file in txt_files:
        str_video_id = (txt_file.split("/")[-1]).split(".")[0]
        video_id = int(str_video_id) * 10000
        with open(txt_file, 'r') as file:
            for line in file:
                
                # Split the line into individual fields
                fields = line.strip().split()
                # Parse the fields into their respective types
                image_id = video_id + int(fields[0])
                object_id = int(fields[1])
                class_id = int(fields[2])
                height = int(fields[3])
                width = int(fields[4])
                bin_rle = fields[5]
                image_path = path_images + str_video_id + "/" + image_str_gen(int(fields[0]))
                # print(f"Image path {image_path}")
                rle = {
                    'counts': bin_rle,  # The RLE counts
                    'size': [height, width]  # Size of the mask (height, width)
                }
                binary_mask = mask.decode(rle)
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                x, y, w, h = cv2.boundingRect(contours[0])
                class_id = class_id if class_id != 10 else 3
                if class_id < 3:
                    class_id = class_id - 1 
                    bbox = [x, y, w, h]
                    if not image_id in count_detections:
                        count_detections[image_id] = 0
                    count_detections[image_id] += 1
                    json_dict["annotations"].append({ "id": counter, "image_id": image_id, "category_id": class_id, "bbox": bbox, "area": w*h, "segmentation":rle, "iscrowd": 0 })
                    if not image_id in seen_images:
                        json_dict["images"].append({ "id": image_id, "width": width, "height": height, "image":image_path })
                        seen_images.add(image_id)
                    counter += 1
    
    print(f"Max detection: {max(count_detections.values())}")
    with open(output_json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return output_json_path

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', required=False, default="/ghome/c5mcv05/team5_split_KITTI-MOTS/", help='Testing images')
    parser.add_argument('--gt_path', required=False, default="/ghome/c5mcv05/team5_split_KITTI-MOTS/instances_txt/", help='GT of testing image')
    parser.add_argument('--eval_gt_json', required=False, default="./eval_gt.json", help='Evaluation gt json path')
    parser.add_argument('--train_gt_json', required=False, default="./train_gt.json", help='Training gt json path')
    args = parser.parse_args()

    strs = ("eval/", "train/")
    load_gt(args.gt_path+strs[0], args.eval_gt_json, args.images_path+strs[0])
    load_gt(args.gt_path+strs[1], args.train_gt_json, args.images_path+strs[1])