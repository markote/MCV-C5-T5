#!/usr/bin/env python3
import os
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
import random
import imageio

# Create a cache for unique class colors
color_cache = {}
def get_color_for_class(cls_id: int):
    cls_id = cls_id.item()
    if cls_id not in color_cache:
        random.seed(cls_id)
        color = tuple(random.randint(0, 255) for _ in range(3))
        color_cache[cls_id] = color
    return color_cache[cls_id]

yl = "11n"
print(f"yolo{yl}.pt")
yln = f"finetuned"

# Define the directory for evaluation images.
eval_images_dir = "my_datasets/KITTI_MOTS_YOLO/test/0018"

# Load the fine-tuned YOLO model (adjust the path as needed)
model = YOLO("/mnt/home/MCV-C5/yolo/runs/detect/kitti_mots_finetune/weights/best.pt")

# Get list of all PNG images from the evaluation folder (filtering on a specific pattern, e.g., "0018*.png")
image_paths = glob.glob(os.path.join(eval_images_dir, "*.png"))
if not image_paths:
    print(f"No images found in {eval_images_dir}")
    exit(1)

frames = []  # This list will store processed frames for the GIF

# Run inference on each image and process the predictions.
for image_path in image_paths:
    img = cv2.imread(image_path)
    results = model.predict(image_path, device='cuda:0')
    for result in results:
        # Get bounding boxes, confidence scores, and class IDs.
        boxes = result.boxes.xyxy.cpu().numpy()      # Coordinates in xyxy format
        confs = result.boxes.conf.cpu().numpy()        # Confidence scores
        clss = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label_text = f"{model.model.names[cls_id]} {conf:.2f}"
            
            # Get a unique color for this class ID.
            color = get_color_for_class(cls_id)
            
            # Draw the bounding box.
            cv2.rectangle(
                img, 
                (x1, y1), (x2, y2), 
                color=color, 
                thickness=1
            )
            
            # Calculate the text size.
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                thickness=1
            )
            
            # Define the text box coordinates (placing it above the bounding box).
            text_x1, text_y1 = x1, y1
            text_x2, text_y2 = x1 + text_width, y1 - text_height - baseline
            
            # Draw a filled rectangle for the label background.
            cv2.rectangle(
                img, 
                (text_x1, text_y1), (text_x2, text_y2), 
                color=color, 
                thickness=-1  # Filled rectangle.
            )
            
            # Put the label text (in black) on top of the filled rectangle.
            cv2.putText(
                img, 
                label_text, 
                (x1, y1 - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0),  # Black text.
                thickness=1
            )
    
    # Convert BGR to RGB (imageio expects RGB) and add the processed image to frames.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(img_rgb)
    print(f"Processed: {image_path}")

# Save all frames as an animated GIF.
output_gif_path = "predictions.gif"
# Duration per frame (in seconds), adjust as needed.
imageio.mimsave(output_gif_path, frames, duration=0.1)
print(f"GIF saved to '{output_gif_path}'")