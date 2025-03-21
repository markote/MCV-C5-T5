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

def draw_predictions(image_path, predictions, output_path, id2label, remove=True, gt_struct=None):
    
    # Load the image
    image = cv2.imread(image_path)
    frame_id = int ((image_path.split("/")[-1]).split(".")[0])
    video_id = int ((image_path.split("/")[-2])) * 10000
    image_id = frame_id + video_id

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
        if label in id2label and (not remove or id2label[label] != 'Misc.'):
            # Draw the bounding box (blue color)
            if id2label[label] == 'car':
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
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

def load_gt(gt_path, output_json_path="./gt.json"):
    pattern = os.path.join(gt_path, '*.txt')
    txt_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Json gt: {txt_files}")
    json_dict = {}
    json_dict["images"] = []
    json_dict["annotations"] = []
    json_dict["categories"] = [{ "id": 1, "name": "car" }, { "id": 2, "name": "pedestrian" }, { "id":252 , "name": "miscelaneous" } ]
    seen_images = set()
    counter = 0
    for txt_file in txt_files:
        video_id = int((txt_file.split("/")[-1]).split(".")[0]) * 10000
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
                rle = {
                    'counts': bin_rle,  # The RLE counts
                    'size': [height, width]  # Size of the mask (height, width)
                }
                binary_mask = mask.decode(rle)
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                x, y, w, h = cv2.boundingRect(contours[0])
                class_id = class_id if class_id !=10 else 1 # replace van anot to car anot
                bbox = [x, y, w, h]
                json_dict["annotations"].append({ "id": counter, "image_id": image_id, "category_id": class_id, "bbox": bbox, "area": w*h, "iscrowd": 0 })
                if not image_id in seen_images:
                    json_dict["images"].append({ "id": image_id, "width": width, "height": height })
                    seen_images.add(image_id)
                counter += 1
    
    with open(output_json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)
    
    return output_json_path

def create_mapping(org_mapping):
    # map coco classes to KITTI-MOTTS 1 Car, 2 Pedestrian, 10 Van and 252 Misc.
    #id2label_mapping = {0: 'N/A', 1: 'person', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 2: 'bicycle', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 3: 'car', 30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 4: 'motorcycle', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 5: 'airplane', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 6: 'bus', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 7: 'train', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 8: 'truck', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 9: 'boat', 90: 'toothbrush'}
    id2label_mapping = {x:'Misc.' if x!=1 and x!=3 and x!=6 and x!=8 else 'pedestrian' if x==1 else 'car' for x in sorted(org_mapping.keys())}
    id2id_mapping =  {x:252 if x!=1 and x!=3 and x!=6 and x!=8 else 1 if x==1 else 0 for x in sorted(org_mapping.keys())}

    return id2label_mapping,id2id_mapping

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

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=False, default="../../team5_split_KITTI-MOTS/eval/", help='Testing images')
    parser.add_argument('--gt_path', required=False, default="../../team5_split_KITTI-MOTS/instances_txt/eval/", help='GT of testing image')
    parser.add_argument('--json_output', required=False, default="./inference_pretrain_DETR.json", help='Output xml')
    parser.add_argument('--gt_json_path', required=False, default="./eval_gt.json", help='gt json path')
    parser.add_argument('--draw', action='store_true', help="Enable drawing predictions")
    parser.add_argument('--metric', action='store_true', help="Enable metric computation")
    parser.add_argument('--charge_gt', action='store_true', help="Enable metric computation")
    args = parser.parse_args()
    print(f"PAth im: {args.test_path}")
    array_images = subimage_gen(args.test_path)
    split_images = split_array_by_size(array_images, 20)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    id2label_kitti_motts_mapping, id2id_kitti_motts_mapping = create_mapping(model.config.id2label)

    # print(f"id2label_kitti_motts_mapping: {id2label_kitti_motts_mapping}")
    # print(f"id2id_kitti_motts_mapping: {id2id_kitti_motts_mapping}")

    j = 0
    if args.metric:
        map_label = id2label_kitti_motts_mapping
        gt_struct = process_coco_json(args.gt_json_path)
        if args.charge_gt:
           load_gt(args.gt_path, args.gt_json_path)
    else:
        map_label = id2label_kitti_motts_mapping
        gt_struct = None

    json_predict = []
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
            if args.metric :
                increase_json(json_predict, array_images[j], im_result, id2id_kitti_motts_mapping)
            if args.draw:
                new_path = array_images[j]
                new_path = new_path.replace(args.test_path, "./eval_prediction_pretrain_DETR/", 1) 
                draw_predictions(array_images[j], im_result, new_path, map_label, gt_struct=gt_struct)
            j += 1
        
        print(f"End split {i}")
    
    if args.metric:
        with open(args.json_output, 'w') as json_file:
            json.dump(json_predict, json_file, indent=4) 
        compute_metric(args.gt_json_path, json_predict)
