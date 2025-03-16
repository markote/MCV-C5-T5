from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def load_predictions(pred_json):
    """
    Extracts and reformats the prediction data from the custom structure into COCO's expected format.
    
    Args:
        pred_json (str): Path to the predicted JSON file.

    Returns:
        list: List of predictions in the correct COCO format.
    """
    with open(pred_json, 'r') as f:
        data = json.load(f)
    
    # Check the structure of predictions (for debugging)
    print("Structure of predictions data:", data.get('annotations', None))

    predictions = []
    
    # Assuming predictions are inside a list under 'predictions' key
    for prediction in data.get('annotations', []):
        # Check the type of 'prediction' to debug the issue
        if isinstance(prediction, dict):
            coco_pred = {
                'image_id': prediction['image_id'],
                'category_id': prediction['category_id'],
                'bbox': prediction['bbox'],  # [x, y, width, height]
                'score': prediction['score']
            }
            if 'segmentation' in prediction:
                coco_pred['segmentation'] = prediction['segmentation']
            predictions.append(coco_pred)
        else:
            print("Skipping invalid prediction:", prediction)

    return predictions

def evaluate_coco(gt_json, pred_json):
    """
    Evaluates the mAP between ground truth and predicted COCO JSON files for both bounding boxes and segmentation.

    Args:
        gt_json (str): Path to the ground truth COCO JSON file.
        pred_json (str): Path to the predicted COCO JSON file.

    Returns:
        dict: COCO evaluation results including mAP@IoU=[0.50:0.95], mAP@50, and mAP@75 for bbox and segm.
    """
    # Load ground truth
    coco_gt = COCO(gt_json)
    
    # Load and reformat predictions
    coco_dt = coco_gt.loadRes(load_predictions(pred_json))

    results = {}
    
    for iou_type in ["bbox", "segm"]:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results[f"mAP_50:95_{iou_type}"] = coco_eval.stats[0]  # mAP@[0.50:0.95]
        results[f"mAP_50_{iou_type}"] = coco_eval.stats[1]     # mAP@50
        results[f"mAP_75_{iou_type}"] = coco_eval.stats[2]     # mAP@75
    
    return results

# Paths to JSON files
gt_json = "eval_gt2.json"   # Path to COCO ground truth
pred_json = "output_predictions_inference2.json"  # Path to COCO predictions

# Run evaluation
results = evaluate_coco(gt_json, pred_json)
print("\nEvaluation Results:", results)