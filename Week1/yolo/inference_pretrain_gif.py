#!/usr/bin/env python3
import os
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
import random
import imageio

color_cache = {}
def get_color_for_class(cls_id: int):
    cls_id = cls_id.item()
    if cls_id not in color_cache:
        random.seed(cls_id)
        color = tuple(random.randint(0, 255) for _ in range(3))
        color_cache[cls_id] = color
    return color_cache[cls_id]

yl = "11x"
# Define the directories.
eval_images_dir = "my_datasets/KITTI_MOTS_YOLO/test/0018"
prediction_dir = f"prediction_{yl}"
# (The directory creation is no longer needed for saving images, but kept if required for other uses)
Path(eval_images_dir.replace("my_datasets/KITTI_MOTS_YOLO", prediction_dir)).mkdir(parents=True, exist_ok=True)

print(f"yolo{yl}.pt")
yln = f"yolo{yl}.pt"
model = YOLO(yln)

# Override the internal class names.
model.model.names = {
    0: "person",
    1: "car",
    **{i: f"class_{i}" for i in range(2, 80)}
}

# Get list of all PNG images from the evaluation folder.
image_paths = glob.glob(os.path.join(eval_images_dir, "*.png"))
if not image_paths:
    print(f"No images found in {eval_images_dir}")
    exit(1)

frames = []  # List to collect processed frames for the GIF

# Run inference on each image and process the predictions.
for image_path in image_paths:
    img = cv2.imread(image_path)
    results = model.predict(image_path, device='cuda:0')
    for result in results:
        # Get bounding boxes, confidence scores, and class IDs.
        boxes = result.boxes.xyxy.cpu().numpy()      # Bounding box coordinates (xyxy)
        confs = result.boxes.conf.cpu().numpy()        # Confidence scores
        clss = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        coco2kittimots = {2.: 1., 5.: 1., 7.: 1., 0: 0.}
        for i, c in enumerate(clss):
            if c.item() in coco2kittimots.keys():
                clss[i] = coco2kittimots[c.item()]
            else:
                clss[i] = 79.

        for box, conf, cls_id in zip(boxes, confs, clss):
            # Only process boxes with class IDs mapped to coco2kittimots values.
            if cls_id.item() in coco2kittimots.values():
                x1, y1, x2, y2 = map(int, box)
                label_text = f"{model.model.names[cls_id]} {conf:.2f}"
                color = get_color_for_class(cls_id)

                # Draw bounding box.
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1)

                # Get text size and compute coordinates for the label background.
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1
                )
                text_x1, text_y1 = x1, y1
                text_x2, text_y2 = x1 + text_width, y1 - text_height - baseline

                # Draw filled rectangle for text background.
                cv2.rectangle(
                    img, 
                    (text_x1, text_y1), (text_x2, text_y2), 
                    color=color, 
                    thickness=-1
                )

                # Put the label text (in black) on top of the rectangle.
                cv2.putText(
                    img, 
                    label_text, 
                    (x1, y1 - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0),
                    thickness=1
                )
    print(f"Processed: {image_path}")
    
    # Convert BGR to RGB as imageio expects RGB images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frames.append(img_rgb)

# Save the collected frames as an animated GIF.
output_gif_path = "predictions_pretrain.gif"
imageio.mimsave(output_gif_path, frames, duration=0.1)
print(f"GIF saved to '{output_gif_path}'")