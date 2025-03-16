import cv2
import numpy as np
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import time
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
            print(f"Error opening image {path_im}: {e}")
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


def draw_segments(im_path, masks, colors):
    None


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
                binary_mask = mask_utils.decode(rle)
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


def map_to_kitti(id_old):
    #{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable', 61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    if id_old == 0:
        return ('pedestrian', 1)
    elif id_old == 2 or id_old == 5 or id_old == 7:  # car bus and truck
        return ('car', 0)
    else:
        return ('Misc.', 252)



def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)

  return visual_mask


def draw_segments(im_path, masks, colors, output=".", masks_gt=None):
    # Load the image
    image = cv2.imread(im_path)
    if image is None:
        raise ValueError(f"Error loading image: {im_path}")
    
    alpha = 0.5  # Transparency factor
    mask_filled = np.zeros_like(image, dtype=np.uint8)

    # Draw contours for each mask
    for mask, color in zip(masks, colors):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, thickness=2)  # Draw contours

        # Create filled transparent mask
        cv2.drawContours(mask_filled, contours, -1, color, thickness=cv2.FILLED)
    
    if masks_gt is not None:
        for gt_mask in masks_gt:
            contours_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours_gt, -1, (0, 255, 0), thickness=2)  # Green, thinner line
    
    # Blend the overlay with the original image
    cv2.addWeighted(mask_filled, alpha, image, 1 - alpha, 0, image)

    # Extract last two parts of the path
    parts = im_path.split("/")[-2:]
    new_path = os.path.join(output, parts[0][-3:], parts[1])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    # Save the new image
    cv2.imwrite(new_path, image)
    print(f"Saved segmented image at: {new_path}")



def load_gt(coco_json_path):
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    masks_dict = {}
    
    # Create a mapping of image_id to image dimensions
    image_info = {img["id"]: (img["width"], img["height"]) for img in coco_data["images"]}
    
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        width, height = image_info[image_id]
        
        # Decode the mask (RLE or polygon-based)
        if "segmentation" in ann:
            mask = mask_utils.decode(ann["segmentation"]).astype(np.uint8)
        else:
            continue
        
        # Ensure masks_dict entry exists
        if image_id not in masks_dict:
            masks_dict[image_id] = []
        
        # Append mask
        masks_dict[image_id].append(mask)
    
    return masks_dict

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=False, default="../../team5_split_KITTI-MOTS/eval/", help='Testing images')
    parser.add_argument('--gt_path', required=False, default="../../team5_split_KITTI-MOTS/instances_txt/eval/", help='GT of testing image')
    parser.add_argument('--json_output', required=False, default="./inference_pretrain_mask2former.json", help='Output xml')
    parser.add_argument('--gt_json_path', required=False, default="./eval_gt.json", help='gt json path')
    parser.add_argument('--checkpoint', required=False, default="facebook/mask2former-swin-small-coco-instance", help="Checkpoint to test")
    parser.add_argument('--output_drawings', required=False, default="./pretrain_eval/", help="Checkpoint to test")
    parser.add_argument('--draw', action='store_true', help="Enable drawing predictions")
    parser.add_argument('--metric', action='store_true', help="Enable metric computation")
    parser.add_argument('--finetune', action='store_true', help="The model in checkpoint is finetuned or not")

    args = parser.parse_args()
    # print(f"PAth im: {args.test_path}")
    array_images = subimage_gen(args.test_path)
    split_images = split_array_by_size(array_images, 4)
    processor = Mask2FormerImageProcessor.from_pretrained(args.checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # id2label_kitti_motts_mapping, id2id_kitti_motts_mapping = create_mapping(model.config.id2label)

    # # print(f"id2label_kitti_motts_mapping: {id2label_kitti_motts_mapping}")
    # # print(f"id2id_kitti_motts_mapping: {id2id_kitti_motts_mapping}")

    if args.metric:
        gt = load_gt(args.gt_json_path)
    else:
        gt = None

    predictions_json = []
    times = []
    for i, split in enumerate(split_images):
        print(f"Begin split {i}")
        path_image, images, images_size = load_images(split)
        inputs = processor(images=images, return_tensors="pt")# need to pass the annotaion in a format DETR on COCO expects, so class number between 0 and 79 for metric evaluation
        

        start_time = time.time()  # Start time

        inputs = inputs.to(device)
        outputs = model(**inputs)
                
        end_time = time.time()  # End time
        execution_time = (end_time - start_time)/len(images)
        times.append(execution_time)
        # print("----")
        # print(results)

        target_sizes = [ (im.shape[::-1][2], im.shape[::-1][1]) for im in images]
        results = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)
        for j, im_result in enumerate(results):
            im_path = path_image[j]
            frame_id = int ((im_path.split("/")[-1]).split(".")[0])
            video_id = int ((im_path.split("/")[-2])) * 10000
            image_id = frame_id + video_id
            masks = []
            colors = []
            for segment in im_result['segments_info']:
                #print("Visualizing mask for instance:", model.config.id2label[segment['label_id']])
                mask = get_mask(im_result['segmentation'], segment['id'])
                mask[mask>0] = 1
                rle = mask_utils.encode(np.asfortranarray(mask))
                #Convert bytes to a string for JSON serialization
                rle['counts'] = rle['counts'].decode('utf-8')
                label = segment["label_id"]
                if not args.finetune:
                    name_label, label = map_to_kitti(label)
                    if label != 252:
                        prediction = {
                            "image_id" : image_id,
                            "score": segment["score"],
                            "category_id": label,
                            "segmentation": rle,  # rle
                        }
                        predictions_json.append(prediction)
                        masks.append(mask)
                        colors.append((0,0,255) if label == 0 else (0,255,255) )
                else:  
                    prediction = {
                            "image_id" : image_id,
                            "score": segment["score"],
                            "category_id": label,
                            "segmentation": rle,  # rle
                    }
                    predictions_json.append(prediction)
                    masks.append(mask)
                    colors.append((0,0,255) if label == 0 else (0,255,255) )

            if args.draw:
                draw_segments(im_path, masks, colors, output=args.output_drawings, masks_gt=gt[image_id] if gt is not None and image_id in gt else None)
        
        print(f"End split {i}")
        # if i > 3:
        #     break
    
    if args.metric:
        with open(args.json_output, 'w') as json_file:
            json.dump(predictions_json, json_file, indent=4) 
        compute_metric(args.gt_json_path, predictions_json)
    
    times = np.array(times)
    print(f"Time per image, avg: ",np.mean(times) ,"std: ",np.std(times))

    