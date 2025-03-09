import argparse
import json
import os
import torch
import numpy as np
import cv2
import sys
import glob
import os

from datasets import load_dataset, Dataset, DatasetDict
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer, AutoImageProcessor
from PIL import Image
from functools import partial
import albumentations as A
from pycocotools.coco import COCO
from pycocotools import mask
from pycocotools.cocoeval import COCOeval

from transformers import AutoModelForObjectDetection
from transformers.image_transforms import center_to_corners_format
from transformers import EarlyStoppingCallback

from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision



@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


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
            print(f"Error opening image: {e}")
            sys.exit(1)

    return full_array_path, result, result_size

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

def increase_json(json_predict, path_im, predictions):
    scores = predictions['scores']
    boxes = predictions['boxes']
    labels = predictions['labels']
    frame_id = int ((path_im.split("/")[-1]).split(".")[0])
    video_id = int ((path_im.split("/")[-2])) * 10000
    image_id = frame_id + video_id
    for i, score in enumerate(scores):
        label = labels[i].item()
        box = boxes[i]
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        bbox = [x_min.item(), y_min.item(), w.item(), h.item()]
        # print(f"boxes x,y,w,h {type(bbox[0]),type(bbox[1]),type(bbox[2]),type(bbox[3])}")
        # print(f"score: {type(score)}")
        json_predict.append({ "image_id":image_id, "category_id": label, "bbox": bbox, "score": score.item() })

def split_array_by_size(arr, partition_size):
    partitions = [arr[i:i + partition_size] for i in range(0, len(arr), partition_size)]
    return partitions

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

def load_json(path_json):
    try:
        with open(path_json, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{path_json}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in file '{path_json}'.")
    return None

def process_json(path_json):
    json_struct = load_json(path_json)
    id2label = {cat["id"]: cat["name"] for cat in json_struct["categories"] if cat["id"]!=252}
    label2id = {cat["name"]: cat["id"] for cat in json_struct["categories"] if cat["id"]!=252}
    images =  {img["id"]: img for img in json_struct["images"]}
    width = 1242
    height = 375

    result = dict()
    for anot in json_struct["annotations"]:
        image_id = anot["image_id"]
        image_path = images[image_id]["image"]
        if not image_id in result:
            result[image_id] = {
                "image_id": image_id,
                "image_path": image_path, # absolut path
                "bbox": [],
                "category_id": [],
                "area": []
                }
        
        result[image_id]["bbox"].append(anot["bbox"]) # COCO format: [x_min, y_min, width, height]
        result[image_id]["category_id"].append(anot["category_id"])
        result[image_id]["area"].append(anot["area"])
    
    data = [result[k] for k in sorted(result.keys())]

    return width, height, id2label,label2id,images,Dataset.from_list(data)

def data_aug(train=True):
    train_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"]
        ),
    )

    val_transform = A.Compose(
        [
            A.NoOp(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"]
        ),
    )
    if train:
        return train_transform
    else:
        return val_transform

def transform_and_augmentation(examples, data_aug, image_processor, return_pixel_mask=False):
    
    images = []
    annotations = []
    #print(f"Type examples: {type(examples)}")
    # e1 = len(examples["image_id"])
    # e2 = len(examples["image_path"])
    # e3 = len(examples["bbox"])
    # e4 = len(examples["category_id"])
    # e5 = len(examples["area"])
    # print(f"Number of examples: {e1,e2,e3,e4,e5}")

    for image_id, path, bboxes, categories, areas in zip(
                                                        examples["image_id"],
                                                        examples["image_path"],
                                                        examples["bbox"],
                                                        examples["category_id"],
                                                        examples["area"]):
        image = Image.open(path).convert("RGB")
        out = data_aug(image=np.array(image), bboxes=bboxes, class_labels=categories)

        data_aug_bboxes = out["bboxes"] # should be COCO format
        data_aug_categories = out["class_labels"]

        formatted_anot = [
            {"bbox": data_aug_bboxes[i], 
            "category_id": data_aug_categories[i], 
            "area":(data_aug_bboxes[i][2] * data_aug_bboxes[i][3]), 
            "is_crowd":0} 
            for i in range(len(data_aug_bboxes))]

        annotations.append({"image_id": image_id, "annotations": formatted_anot})
        images.append(out["image"])
    #print(f"Annotations: {annotations}")
    encoding = image_processor(images=images, annotations=annotations, return_tensors="pt")
    # print(f"Anots:{annotations}")
    # print(f"Encoding:{encoding}")
    if not return_pixel_mask:
        encoding.pop("pixel_mask", None)
    return encoding
    # return {
        # "pixel_values": encoding["pixel_values"],  # Required by DETR
        # "labels": encoding["labels"]  # Required by DETR
    # }

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

def training(model, dataset, output_dir=f"./detr_finetuned", per_device_train_batch_size=2, per_device_eval_batch_size=2, num_train_epochs=10, save_strategy="epoch", evaluation_strategy="epoch", logging_dir="./logs", learning_rate=5e-5, weight_decay=0.01, save_total_limit=2, report_to="none"):
    # Training arguments
    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        num_train_epochs = num_train_epochs,
        save_strategy = save_strategy,
        #evaluation_strategy = evaluation_strategy,
        logging_dir = logging_dir,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        save_total_limit = save_total_limit, # Keep only last 2 checkpoints
        report_to = report_to,     # Avoid logging to external services
        remove_unused_columns=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

def save_sample_opencv(encoding, idx=0, filename="sample.png"):
    # Unnormalize the image (assumes ImageNet statistics)
    image_tensor = encoding["pixel_values"][idx]
    image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = image_tensor * image_std + image_mean
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)

    # Convert from float [0,1] to uint8 [0,255] and BGR (OpenCV format)
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # The labels for each sample are in a dictionary; expect key "boxes" containing [x_min, y_min, x_max, y_max]
    sample_labels = encoding["labels"][idx]
    boxes = sample_labels["boxes"]  # Adjust this key if needed
    height, width, _ = image_np.shape
    for box in boxes:
        #print(f"BOXES: {box}")
        
        x_center, y_center, w, h = map(float, box)  # Convert to integers
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

    # Save the image
    save_path = os.path.join("./", filename)
    cv2.imwrite(save_path, image_np)
    print(f"Saved image: {save_path}")  
    
def eval(model, processor):
    json_output = "./inference_finetune_DETR.json"
    eval_gt_path = "./eval_gt.json"
    eval_path = "/ghome/c5mcv05/team5_split_KITTI-MOTS/eval/"
    output_dir= "detr_finetuned_kt_mt_best_from_trainer/"
    
    array_images = subimage_gen(eval_path)
    split_images = split_array_by_size(array_images, 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    json_predict = []
    j = 0
    for i, split in enumerate(split_images):
        print(f"Begin split {i}")
        path_image, images, images_size = load_images(split)
        inputs = processor(images=images, return_tensors="pt").to(device)# need to pass the annotaion in a format DETR on COCO expects, so class number between 0 and 79 for metric evaluation
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor(images_size) #[image.size[::-1]]
        # print(f"TArget size: {target_sizes}")
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes) #, threshold=0.8) increase shit

        #save_coco_json(im_result, json_output)
        for im_result in results:
            increase_json(json_predict, array_images[j], im_result)
            j += 1
        
        print(f"End split {i}")
    
    with open(json_output, 'w') as json_file:
        json.dump(json_predict, json_file, indent=4)
    compute_metric(eval_gt_path, json_predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_output', required=False, default="inference_finetune_DETR.json", help='Output xml')
    parser.add_argument('--eval_gt_json_path', required=False, default="./eval_gt.json", help='Evaluation gt json path')
    parser.add_argument('--train_gt_json_path', required=False, default="./train_gt.json", help='Training gt json path')
    args = parser.parse_args()

    checkpoint = "facebook/detr-resnet-50"
    _,_,id2label,label2id,_,train_dataset = process_json(args.train_gt_json_path)
    _,_,e_id2label,e_label2id,_,eval_dataset = process_json(args.eval_gt_json_path)
    print(f"id2label: {id2label}")
    print(f"label2id: {label2id}")
    # print(f"dataset. {train_dataset[149:150]}")
    # max_bbox = 0
    # for t in train_dataset:
    #     if len(t["bbox"]) > max_bbox:
    #         max_bbox = len(t["bbox"])

    # for e in eval_dataset:
    #     if len(e["bbox"]) > max_bbox:
    #         max_bbox = len(e["bbox"])
    #IMAGENET std and mean, standard things
    # image_mean = [0.485, 0.456, 0.406]
    # image_std = [0.229, 0.224, 0.225]
    # print(f"max detection picture: {max_bbox}")
    IMAGE_SIZE = 480

    MAX_SIZE = IMAGE_SIZE

    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
        do_pad=True,
        pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
    )
    
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    print(f"Images on train: {len(train_dataset)}")
    train_anots = sum([len(x["bbox"]) for x in train_dataset])
    print(f"Total anot on train: {train_anots}")

    print(f"Images on eval: {len(eval_dataset)}")
    train_eval = sum([len(x["bbox"]) for x in eval_dataset])
    print(f"Total anot on eval: {train_eval}")

    print("Model is on:", next(model.parameters()).device)
    train_data_aug = data_aug(train=True)
    train_transform = partial(transform_and_augmentation, data_aug=train_data_aug, image_processor=image_processor)
    train_dataset = train_dataset.with_transform(train_transform)
    # transformed_sample = train_transform(train_dataset[149:150])
    # save_sample_opencv(transformed_sample, idx=0, filename="debug_sample_0.png")


    eval_data_aug = data_aug(train=False)
    eval_transform = partial(transform_and_augmentation, data_aug=eval_data_aug, image_processor=image_processor)
    eval_dataset = eval_dataset.with_transform(eval_transform)

    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    training_args = TrainingArguments(
        output_dir="detr_finetuned_kt_mt",
        num_train_epochs=60,
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to= "none",
        log_level="debug",
        save_steps=None,  # Ensure it doesn't save by steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    print("Trainer will use:", training_args.device)
    trainer.train()
    print("Eval last trained model")
    m = trainer.evaluate()  # This will evaluate the final loaded model
    print(f"Metrics {m}")
    trainer.save_model("detr_finetuned_kt_mt_best_from_trainer")  # Model is saved to this directory
    trainer.save_state()
    model.save_pretrained("detr_finetuned_kt_mt_best")
    image_processor.save_pretrained("detr_finetuned_kt_mt_best")
    print("END")
    eval(model, image_processor)