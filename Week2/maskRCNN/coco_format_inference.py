import torch
import os
import json
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# COCO class to KITTI-MOTS mapping
COCO_TO_CUSTOM = {
    1: "pedestrian",
    3: "car",
    4: "car",
    6: "car"
}

CATEGORY_ID_MAPPING = {
    "car": 1,
    "pedestrian": 2
}

categories = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "pedestrian", "supercategory": "person"}
]

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def create_coco_format_output(coco_json_path, images_dir, output_coco_json, predictor):
    if os.path.dirname(output_coco_json):
        os.makedirs(os.path.dirname(output_coco_json), exist_ok=True)

    
    coco = COCO(coco_json_path)
    image_ids = coco.getImgIds()

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    for image_id in tqdm(image_ids, desc="Processing images"):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(images_dir, image_info["file_name"])

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping.")
            continue

        image = cv2.imread(image_path)
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_info["file_name"],
            "height": image_info["height"],
            "width": image_info["width"]
        })

        for i in range(len(instances)):
            box = instances.pred_boxes.tensor[i].numpy()
            original_label = int(instances.pred_classes[i].numpy()) + 1  # COCO 1-based index
            score = float(instances.scores[i].numpy())

            if original_label in COCO_TO_CUSTOM and score > 0.5:
                mapped_class = COCO_TO_CUSTOM[original_label]
                category_id = CATEGORY_ID_MAPPING[mapped_class]
                
                bbox = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
                area = bbox[2] * bbox[3]
                
                mask = None
                if instances.has("pred_masks"):
                    mask = instances.pred_masks[i].numpy().astype(np.uint8)
                    mask_rle = mask_utils.encode(np.asfortranarray(mask))
                    mask_rle["counts"] = mask_rle["counts"].decode("utf-8")  # JSON compatibility
                
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "score": score,
                    "segmentation": mask_rle if mask is not None else []
                }
                coco_output["annotations"].append(ann)
                annotation_id += 1

    with open(output_coco_json, "w") as json_file:
        json.dump(coco_output, json_file, indent=4)
    
def evaluate_coco(gt_coco_json, pred_coco_json):
    from detectron2.data.datasets import load_coco_json
    
    dataset_name = "coco_eval"
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(gt_coco_json, "", dataset_name))
    MetadataCatalog.get(dataset_name).set(thing_classes=["car", "pedestrian"])
    
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir="./coco_eval_results")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    COCO_JSON_PATH = "eval_gt2.json"
    IMAGES_DIR = "eval_photos"
    OUTPUT_COCO_JSON = "output_predictions_inference.json"
    RUN_EVAL = True

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    create_coco_format_output(COCO_JSON_PATH, IMAGES_DIR, OUTPUT_COCO_JSON, predictor)

    if RUN_EVAL:
        evaluate_coco(COCO_JSON_PATH, OUTPUT_COCO_JSON)
