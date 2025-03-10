#!/usr/bin/env python3
import os
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
import random 

color_cache = {}
def get_color_for_class(cls_id: int):
    cls_id = cls_id.item()
    if cls_id not in color_cache:
        random.seed(cls_id)
        color = tuple(random.randint(0, 255) for _ in range(3))
        color_cache[cls_id] = color
    return color_cache[cls_id]

yl = "11n"
# Define the directories.
eval_images_dir = "my_datasets/KITTI_MOTS_YOLO/eval/images"
prediction_dir = f"prediction_{yl}"
Path(eval_images_dir.replace("my_datasets/KITTI_MOTS_YOLO",prediction_dir)).mkdir(parents=True, exist_ok=True)

# Load the pretrained YOLO11x model.
print(f"yolo{yl}.pt")
yln = f"yolo{yl}.pt"
model = YOLO(yln)

# Override the internal class names.
# (Make sure to supply names for all classes up to the highest index used by the model.)
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
image_paths = ["/mnt/home/MCV-C5/yolo/my_datasets/KITTI_MOTS_YOLO/eval/images/0019_000621.png"]
# Run inference on each image and save the predictions.
for image_path in image_paths:
    img = cv2.imread(image_path)
    results = model.predict(image_path, device='cuda:0')
    for result in results:
        # Get the bounding box coordinates, confidence scores, and class IDs.
        # If using CUDA, moving tensors to CPU and converting to numpy arrays.
        boxes = result.boxes.xyxy.cpu().numpy()  # bounding box coordinates in xyxy format
        confs = result.boxes.conf.cpu().numpy()    # confidence scores
        clss = result.boxes.cls.cpu().numpy().astype(int)  # class IDs
        coco2kittimots = {2.:1. , 5.:1., 7.:1., 0:0.}
        for i,c in enumerate(clss):
            if c.item() in coco2kittimots.keys():
                clss[i] = coco2kittimots[c.item()]
            else:
                clss[i] = 79.
        

        for box, conf, cls_id in zip(boxes, confs, clss):
            if cls_id.item() in coco2kittimots.values():
                x1, y1, x2, y2 = map(int, box)
                
                label_text = f"{model.model.names[cls_id]} {conf:.2f}"
                
                # Get a unique color for this class ID
                color = get_color_for_class(cls_id)
                
                # Draw the bounding box
                cv2.rectangle(
                    img, 
                    (x1, y1), (x2, y2), 
                    color=color, 
                    thickness=1
                )
                
                # Put the label text (in black) on a filled rectangle of 'color'
                # 1) First get the text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    thickness=1
                )
                # 2) Compute coordinates for the text box
                #    We'll place it above the top-left corner of the bounding box
                text_x1, text_y1 = x1, y1
                text_x2, text_y2 = x1 + text_width, y1 - text_height - baseline
                
                # 3) Draw a filled rectangle for the text background
                cv2.rectangle(
                    img, 
                    (text_x1, text_y1), (text_x2, text_y2), 
                    color=color, 
                    thickness=-1  # -1 means filled
                )
                
                # 4) Now put the text in black on top of the filled rectangle
                cv2.putText(
                    img, 
                    label_text, 
                    (x1, y1 - baseline), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0),  # black text
                    thickness=1
                )
    print(image_path.replace("my_datasets/KITTI_MOTS_YOLO",prediction_dir))
    cv2.imwrite(image_path.replace("my_datasets/KITTI_MOTS_YOLO",prediction_dir), img)


print(f"Predictions saved to '{prediction_dir}' folder.")