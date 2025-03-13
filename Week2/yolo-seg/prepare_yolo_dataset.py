#!/usr/bin/env python3
import os
import shutil
import cv2
import numpy as np
import argparse
from pycocotools import mask as maskUtils

def create_yolo_substructure(dest_dir, subset="train"):
    """
    Create the YOLO directory structure for a given subset:
      dest_dir/
        <subset>/images/
        <subset>/labels/
    """
    images_dir = os.path.join(dest_dir, subset, "images")
    labels_dir = os.path.join(dest_dir, subset, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir

def parse_annotation_line(line):
    """
    Parse a single line from a KITTI MOTS annotation file.
    Expected format:
       <frame> <track_id> <class_id> <img_height> <img_width> <mask_data>
    """
    parts = line.strip().split(" ")
    if len(parts) < 6:
        return None
    try:
        frame = int(parts[0])
        track_id = int(parts[1])
        class_id = int(parts[2])
        img_height = int(parts[3])
        img_width = int(parts[4])
    except ValueError as e:
        print("Error parsing numeric fields:", e)
        return None
    # The rest of the line is the mask data (as a string)
    mask_data_str = " ".join(parts[5:])
    return frame, track_id, class_id, img_height, img_width, mask_data_str

def process_sequence(
    seq_id, 
    src_img_dir, 
    src_ann_dir, 
    images_dir, 
    labels_dir, 
    class_mapping
):
    """
    Process all frames for a given sequence:
      1. Copy the image to the YOLO images folder (renamed with sequence id).
      2. Parse the corresponding annotations from the instances_txt file.
      3. For each annotation, decode the mask using pycocotools,
         extract the segmentation polygon using contours, normalize the coordinates,
         and write to a label file in the format:
             <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    """
    seq_folder = os.path.join(src_img_dir, seq_id)
    ann_file = os.path.join(src_ann_dir, f"{seq_id}.txt")
    if not os.path.exists(ann_file):
        print(f"Annotation file for sequence {seq_id} not found at {ann_file}.")
        return

    # Group annotations by frame number.
    annotations_by_frame = {}
    with open(ann_file, "r") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is None:
                continue
            frame, track_id, class_id, ann_img_height, ann_img_width, mask_data_str = parsed
            annotations_by_frame.setdefault(frame, []).append(
                (track_id, class_id, ann_img_height, ann_img_width, mask_data_str)
            )

    # Process each frame in the sequence.
    for frame, ann_list in annotations_by_frame.items():
        # Construct the image filename, e.g., "000000.png" for frame 0.
        filename = f"{frame:06d}.png"
        src_image_path = os.path.join(seq_folder, filename)
        if not os.path.exists(src_image_path):
            print(f"Image {src_image_path} not found.")
            continue

        # Load the image to obtain its dimensions.
        img = cv2.imread(src_image_path)
        if img is None:
            print(f"Failed to load image {src_image_path}.")
            continue
        height, width = img.shape[:2]

        # Create a new filename that includes the sequence id.
        new_filename = f"{seq_id}_{filename}"
        dest_image_path = os.path.join(images_dir, new_filename)
        shutil.copy(src_image_path, dest_image_path)

        # Process each annotation for this frame.
        yolo_lines = []
        for track_id, class_id, ann_img_height, ann_img_width, mask_data_str in ann_list:
            # Form the RLE dictionary using the annotation's height and width.
            rle = {"counts": mask_data_str, "size": [ann_img_height, ann_img_width]}
            try:
                binary_mask = maskUtils.decode(rle)
            except Exception as e:
                print(f"Error decoding mask for image {src_image_path}: {e}")
                continue

            # If the annotation's size differs from the loaded image, resize the mask.
            if binary_mask.shape[0] != height or binary_mask.shape[1] != width:
                binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Extract segmentation polygon using contours.
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # Choose the largest contour (if multiple are present)
            polygon = max(contours, key=cv2.contourArea)
            polygon = polygon.flatten()
            # Normalize polygon coordinates to [0, 1]
            normalized_polygon = []
            for i in range(0, len(polygon), 2):
                normalized_polygon.append(polygon[i] / width)
                normalized_polygon.append(polygon[i+1] / height)
            segmentation_str = " ".join(f"{p:.6f}" for p in normalized_polygon)

            # Map the KITTI MOTS class_id to a YOLO class index.
            if class_id in class_mapping:
                yolo_class = class_mapping[class_id]
            else:
                # Skip if the class is not in your mapping.
                continue

            # Annotation format: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
            yolo_line = f"{yolo_class} {segmentation_str}"
            yolo_lines.append(yolo_line)

        # Write segmentation annotation file for this image.
        label_filename = new_filename.replace(".png", ".txt")
        dest_label_path = os.path.join(labels_dir, label_filename)
        with open(dest_label_path, "w") as lf:
            lf.write("\n".join(yolo_lines))

def main(mode):
    # Base directory for your dataset with the new folder structure
    src_base = "/home/msiau/data/tmp/ptruong/data/team5_split_KITTI-MOTS"

    # Destination directory for your YOLO formatted segmentation dataset
    dest_dir = f"/home/msiau/data/tmp/ptruong/data/KITTI_MOTS_YOLO_seg"

    # Create YOLO folder structures for train and eval
    train_images_dir, train_labels_dir = create_yolo_substructure(dest_dir, subset="train")
    eval_images_dir, eval_labels_dir = create_yolo_substructure(dest_dir, subset="eval")

    # Define the class mapping
    # Example: KITTI MOTS uses '1' for Car, '2' for Pedestrian, etc.
    # Here we map them to YOLO class indices.
    class_mapping = {
        "pretrain": {
            1: 1,  # e.g., Car -> YOLO class 1
            2: 0,  # e.g., Pedestrian -> YOLO class 0
        },
        "finetune": {
            1: 1,
            2: 0,
        },
    }

    # Directories containing the train and eval images
    src_train_dir = os.path.join(src_base, "train")  # e.g., 0000, 0001, ...
    src_eval_dir  = os.path.join(src_base, "eval")   # e.g., 0018, 0019, ...

    # Directories containing the train and eval annotations (instances_txt)
    train_ann_dir = os.path.join(src_base, "instances_txt", "train")
    eval_ann_dir  = os.path.join(src_base, "instances_txt", "eval")

    # Process training sequences
    for seq_id in os.listdir(src_train_dir):
        seq_path = os.path.join(src_train_dir, seq_id)
        if os.path.isdir(seq_path):
            print(f"Processing training sequence: {seq_id}")
            process_sequence(
                seq_id, 
                src_train_dir, 
                train_ann_dir, 
                train_images_dir, 
                train_labels_dir, 
                class_mapping[mode]
            )

    # Process evaluation sequences
    for seq_id in os.listdir(src_eval_dir):
        seq_path = os.path.join(src_eval_dir, seq_id)
        if os.path.isdir(seq_path):
            print(f"Processing evaluation sequence: {seq_id}")
            process_sequence(
                seq_id, 
                src_eval_dir, 
                eval_ann_dir, 
                eval_images_dir, 
                eval_labels_dir, 
                class_mapping[mode]
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare segmentation annotations for YOLO')
    parser.add_argument('--mode', type=str, default='pretrain', help='finetune or pretrain')
    args = parser.parse_args()
    main(mode=args.mode)