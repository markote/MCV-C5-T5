import torch
import os
import json
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from pycocotools.coco import COCO

# COCO class to KITTI-MOTS mapping
COCO_TO_CUSTOM = {
    1: "pedestrian",  # Person
    3: "car",         # Car
    4: "car",         # Van
    6: "car"          # Bus
}

# Assign unique category IDs
CATEGORY_ID_MAPPING = {
    "car": 1,
    "pedestrian": 2
}

# Define categories for COCO format
categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "pedestrian", "supercategory": "person"}
]

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def draw_predictions(image, predictions, image_name, save_path):
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        category = pred["category_id"]
        label = next((k for k, v in CATEGORY_ID_MAPPING.items() if v == category), "unknown")

        color = (0, 255, 0) if label == "car" else (0, 0, 255)  # Green for car, Red for bike
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Predictions for {image_name}")
    plt.savefig(os.path.join(save_path, f"vis_{image_name}.png"))
    plt.close()

def create_coco_format_output(coco_json_path, images_dir, output_coco_json, output_raw_json, predictor, save_vis_dir):
    """
    Runs inference on images specified in a COCO JSON annotation file.
    Saves:
    1. The raw Detectron2 predictions.
    2. A converted COCO JSON format with mapped classes.
    3. Visualization of detections every 50 images.
    """
    os.makedirs(save_vis_dir, exist_ok=True) 

    coco = COCO(coco_json_path)
    image_ids = coco.getImgIds()

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    raw_output = {}

    annotation_id = 1

    for idx, image_id in enumerate(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(images_dir, image_info["file_name"])

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping.")
            continue

        image = cv2.imread(image_path)
        outputs = predictor(image)

        raw_output[image_info["file_name"]] = {
            "pred_boxes": outputs["instances"].pred_boxes.tensor.cpu().tolist(),
            "pred_classes": outputs["instances"].pred_classes.cpu().tolist(),
            "scores": outputs["instances"].scores.cpu().tolist()
        }

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_info["file_name"],
            "height": image_info["height"],
            "width": image_info["width"]
        })

        filtered_preds = []
        for i in range(len(outputs["instances"].pred_classes)):
            box = outputs["instances"].pred_boxes.tensor[i].cpu().numpy()
            original_label = int(outputs["instances"].pred_classes[i].cpu().numpy()) + 1  # COCO uses 1-based indexing
            score = float(outputs["instances"].scores[i].cpu().numpy())

            # Map COCO class to custom class
            if original_label in COCO_TO_CUSTOM and score > 0.5:
                mapped_class = COCO_TO_CUSTOM[original_label]
                category_id = CATEGORY_ID_MAPPING[mapped_class]

                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "area": float((box[2] - box[0]) * (box[3] - box[1])),
                    "iscrowd": 0,
                    "score": score
                }
                coco_output["annotations"].append(ann)
                filtered_preds.append(ann)
                annotation_id += 1

        # if idx % 1000 == 0:
        #     draw_predictions(image.copy(), filtered_preds, image_info["file_name"], save_vis_dir)

    # Save results
    with open(output_coco_json, "w") as json_file:
        json.dump(coco_output, json_file, indent=4)

    with open(output_raw_json, "w") as raw_file:
        json.dump(raw_output, raw_file, indent=4)

def evaluate_coco(gt_coco_json, pred_coco_json):
    """
    Evaluate the generated predictions using COCOEvaluator.
    """
    from detectron2.data.datasets import load_coco_json

    dataset_name = "coco_eval"
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(gt_coco_json, "", dataset_name))
    MetadataCatalog.get(dataset_name).set(thing_classes=['car', 'pedestrian', 'miscelaneous'])

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir="./coco_eval_results")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print("Evaluation Results:", results)

if __name__ == "__main__":
    # Hardcoded paths
    COCO_JSON_PATH = "eval_gt_fixed.json"  # Change this to your actual path
    IMAGES_DIR = ""  # Change this to your actual path
    OUTPUT_COCO_JSON = "output_predictions_inference.json"
    OUTPUT_RAW_JSON = "output_raw_inference.json"
    SAVE_VIS_DIR = "output_visualizations"
    RUN_EVAL = True  # Set to False if you don't want to run evaluation

    # Setup Detectron2 predictor
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # Run inference & save results
    create_coco_format_output(COCO_JSON_PATH, IMAGES_DIR, OUTPUT_COCO_JSON, OUTPUT_RAW_JSON, predictor, SAVE_VIS_DIR)

    # Run evaluation if enabled
    if RUN_EVAL:
        evaluate_coco(COCO_JSON_PATH, OUTPUT_COCO_JSON)