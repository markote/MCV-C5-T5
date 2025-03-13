#!/usr/bin/env python3
import os
import random
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import argparse

def load_yolo_segmentation(label_path):
    """
    Reads a YOLO segmentation annotation file where each line is:
       <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    Returns a list of (class_index, polygon),
    where 'polygon' is a numpy array of shape (N, 2) in normalized coordinates.
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                cls_idx = int(parts[0])
            except ValueError:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue
            polygon = np.array(coords).reshape(-1, 2)
            annotations.append((cls_idx, polygon))
    return annotations

def overlay_polygons(image, annotations, color=(0, 255, 0), thickness=2):
    """
    Draws the given list of (class_index, polygon) on the image.
    The polygon coordinates are normalized to [0,1] and are scaled by the image size.
    """
    h, w = image.shape[:2]
    for cls_idx, polygon in annotations:
        pts = polygon.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image

def parse_original_annotation_line(line):
    """
    Parse a single line from an original KITTI MOTS annotation file.
    Expected format:
      <frame> <track_id> <class_id> <img_height> <img_width> <mask_data>
    Returns a tuple:
      (frame, track_id, class_id, ann_img_height, ann_img_width, mask_data_str)
    or None if parsing fails.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    try:
        frame = int(parts[0])
        track_id = int(parts[1])
        class_id = int(parts[2])
        ann_img_height = int(parts[3])
        ann_img_width = int(parts[4])
    except ValueError:
        return None
    mask_data_str = " ".join(parts[5:])
    return frame, track_id, class_id, ann_img_height, ann_img_width, mask_data_str

def main(mode="train"):
    # --- CONFIGURATION ---
    # Processed images and annotations (polygon-based, normalized) from the converted dataset.
    proc_images_dir = "/home/msiau/data/tmp/ptruong/data/KITTI_MOTS_YOLO_seg/{}/images".format(mode)
    proc_labels_dir = "/home/msiau/data/tmp/ptruong/data/KITTI_MOTS_YOLO_seg/{}/labels".format(mode)
    # Original KITTI MOTS annotation files (raw RLE format)
    orig_ann_dir = os.path.join("/home/msiau/data/tmp/ptruong/data", "team5_split_KITTI-MOTS", "instances_txt", mode)
    # Output directory for visualization results
    output_dir = "check_annot_{}".format(mode)
    os.makedirs(output_dir, exist_ok=True)

    # --- SELECT 5 RANDOM IMAGES ---
    image_files = [f for f in os.listdir(proc_images_dir) if f.endswith(".png")]
    if len(image_files) < 5:
        print("Not enough images found in", proc_images_dir)
        return
    selected_images = random.sample(image_files, 5)

    for img_file in selected_images:
        img_path = os.path.join(proc_images_dir, img_file)
        label_file = img_file.replace(".png", ".txt")
        label_path = os.path.join(proc_labels_dir, label_file)

        # Load the original image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_path}")
            continue
        height, width = image.shape[:2]

        # --- PART 1: Overlay Processed Segmentation ---
        annotations = load_yolo_segmentation(label_path)
        overlay_image = image.copy()
        overlay_image = overlay_polygons(overlay_image, annotations)
        overlay_out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(overlay_out_path, overlay_image)
        print(f"Saved overlay image to {overlay_out_path}")

        # --- PART 2: Draw Binary Masks from Raw RLE for Each Object ---
        # Expect filename format: <seq_id>_<frame>.png (e.g., "0002_000123.png")
        if "_" not in img_file:
            print(f"Filename {img_file} does not match expected format <seq_id>_<frame>.png")
            continue
        seq_id, frame_str = img_file.split("_", 1)
        frame_str = frame_str.replace(".png", "")
        try:
            frame_number = int(frame_str)
        except ValueError:
            print(f"Could not parse frame number from {img_file}")
            continue

        # Open the original annotation file for the sequence
        orig_ann_file = os.path.join(orig_ann_dir, f"{seq_id}.txt")
        if not os.path.exists(orig_ann_file):
            print(f"Original annotation file {orig_ann_file} not found.")
            continue

        # Process each line in the original annotation file corresponding to this frame
        with open(orig_ann_file, "r") as f:
            for line in f:
                parsed = parse_original_annotation_line(line)
                if parsed is None:
                    continue
                ann_frame, track_id, class_id, ann_h, ann_w, mask_data_str = parsed
                if ann_frame != frame_number:
                    continue  # only process annotations for this frame

                # Decode the RLE mask
                rle = {"counts": mask_data_str, "size": [ann_h, ann_w]}
                try:
                    decoded_mask = maskUtils.decode(rle)
                except Exception as e:
                    print(f"Error decoding RLE for {img_file}, track_id={track_id}: {e}")
                    continue

                # Resize mask if needed to match the image dimensions
                if decoded_mask.shape[0] != height or decoded_mask.shape[1] != width:
                    decoded_mask = cv2.resize(decoded_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Convert mask to binary image (0 and 255)
                binary_mask = (decoded_mask * 255).astype(np.uint8)

                # Save binary mask for this object
                mask_filename = img_file.replace(".png", f"_obj{track_id}_mask.png")
                mask_out_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_out_path, binary_mask)
                print(f"Saved binary mask for object {track_id} to {mask_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    args = parser.parse_args()
    main(mode=args.mode)