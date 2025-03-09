import torch
import os
import json
import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

# Define COCO Class Mapping
COCO_TO_CUSTOM = {
    # 1: "bike",   # Person
    # 2: "bike",   # Bicycle
    3: "car",    # Car
    # 4: "car",    # Van (assuming COCO's "bus" is a van)
    # 17: "bike"   # Motorcycle
}

# Reverse Mapping to Custom Category IDs
CATEGORY_ID_MAPPING = {
    "car": 1
}

# Define Categories for COCO Format
categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"}
]

def draw_predictions(image, predictions, image_name, save_path):
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        category = pred["category_id"]
        label = "Bike" if category == 1 else "Car"

        color = (0, 255, 0) if label == "Bike" else (255, 0, 0)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Predictions for {image_name}")
    plt.savefig(os.path.join(save_path, f"vis_{image_name}.png"))
    plt.close()

def create_coco_format_output(images_dir, output_coco_json, output_raw_json, predictor, save_vis_dir):
    """
    Runs inference on all images in a directory and saves:
    1. The raw Detectron2 predictions.
    2. A converted COCO JSON with classes mapped to 'bike' or 'car'.
    3. A visualization of the detections every 50 images.
    """
    os.makedirs(save_vis_dir, exist_ok=True) 

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    raw_output = {}

    image_id = 1
    annotation_id = 1

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        outputs = predictor(image)

        raw_output[image_file] = {
            "pred_boxes": outputs["instances"].pred_boxes.tensor.cpu().tolist(),
            "pred_classes": outputs["instances"].pred_classes.cpu().tolist(),
            "scores": outputs["instances"].scores.cpu().tolist()
        }

        # Save image metadata in COCO format
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_file,
            "height": image.shape[0],
            "width": image.shape[1]
        })

        filtered_preds = []
        for i in range(len(outputs["instances"].pred_classes)):
            box = outputs["instances"].pred_boxes.tensor[i].cpu().numpy()
            label = int(outputs["instances"].pred_classes[i].cpu().numpy()) + 1  # COCO class IDs start at 1
            score = float(outputs["instances"].scores[i].cpu().numpy())

            if label in COCO_TO_CUSTOM and score > 0.5:
                mapped_label = COCO_TO_CUSTOM[label]
                mapped_category_id = CATEGORY_ID_MAPPING[mapped_label]

                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": mapped_category_id,
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "area": float((box[2] - box[0]) * (box[3] - box[1])),
                    "iscrowd": 0,
                    "score": score
                }
                coco_output["annotations"].append(ann)
                filtered_preds.append(ann)
                annotation_id += 1

        if idx % 50 == 0:
            draw_predictions(image.copy(), filtered_preds, image_file, save_vis_dir)

        image_id += 1

    # Save COCO format JSON
    with open(output_coco_json, "w") as json_file:
        json.dump(coco_output, json_file, indent=4)

    # Save raw Detectron2 output JSON
    with open(output_raw_json, "w") as raw_file:
        json.dump(raw_output, raw_file, indent=4)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Input and Output Paths
images_dir = "./frames_output"  # Change to your images folder
output_coco_json = "output_predictions.json"  # Processed COCO format JSON
output_raw_json = "output_raw.json"  # Original Detectron2 output
save_vis_dir = "output_visualizations"  # Folder to save visualizations

# Run the function
create_coco_format_output(images_dir, output_coco_json, output_raw_json, predictor, save_vis_dir)
