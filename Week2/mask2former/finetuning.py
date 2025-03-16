import argparse
import json
import os
import torch
import numpy as np
import cv2
import sys
import glob
import os
import time

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

import albumentations as A
from datasets import load_dataset, Dataset, DatasetDict
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

import numpy as np
import cv2
import torch
import random

from torch.utils.data import Dataset as torch_Dataset
from torch.utils.data import DataLoader 
from tqdm.auto import tqdm

class ImageSegmentationDataset(torch_Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, processor, data_aug=None):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.processor = processor
        self.data_aug = data_aug
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.dataset[idx]["image_path"]).convert("RGB"))
        # print(f"shape of image:{image.shape}")
        annotations = []
        id_inst_count = 1
        i_H, i_W, _ = image.shape
        segmentation_map = np.zeros((i_H, i_W))
        inst2class = dict()
        for seg, bbox, category, area in zip(
                                            self.dataset[idx]["segmentation"],
                                            self.dataset[idx]["bbox"],
                                            self.dataset[idx]["category_id"],
                                            self.dataset[idx]["area"]):
                binary_mask = mask_utils.decode(seg)  # Shape (H, W)
                # print(f"Min: {np.min(binary_mask)}")
                # print(f"Max: {np.max(binary_mask)}")
                # cv2.imwrite(f"mask_image_inst{id_inst_count}.png", binary_mask*255 ) 
                segmentation_map[binary_mask == 1] = id_inst_count
                inst2class[np.uint(id_inst_count)] = category
                id_inst_count += 1
        
        if self.data_aug is not None:
            transformed = self.data_aug(image=image, mask=segmentation_map)
            image, segmentation_map = transformed['image'], transformed['mask']

        # convert H, W, C to C, H, W
        #print("shape image before transposing,", image.shape )
        image = np.transpose(image, (2, 0, 1))
        # TODO test without transpose
        inputs = self.processor(images = [image], segmentation_maps = [segmentation_map], ignore_index=0, instance_id_to_semantic_id=inst2class, return_tensors="pt") # do_resize=False,
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        # instance_seg = np.array(self.dataset[idx]["annotation"])[:,:,1]
        # class_id_map = np.array(self.dataset[idx]["annotation"])[:,:,0]
        # class_labels = np.unique(class_id_map)

        # inst2class = {}
        # for label in class_labels:
        #     instance_ids = np.unique(instance_seg[class_id_map == label])
        #     inst2class.update({i: label for i in instance_ids})

        # # apply transforms
        # if self.transform is not None:
        #     transformed = self.transform(image=image, mask=instance_seg)
        #     image, instance_seg = transformed['image'], transformed['mask']
        #     # convert to C, H, W
        #     image = image.transpose(2,0,1)

        # if class_labels.shape[0] == 1 and class_labels[0] == 0:
        #     # Some image does not have annotation (all ignored)
        #     inputs = self.processor([image], return_tensors="pt")
        #     inputs = {k:v.squeeze() for k,v in inputs.items()}
        #     inputs["class_labels"] = torch.tensor([0])
        #     inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        # else:
        #   inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        #   inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs

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
    id2label = {cat["id"]: cat["name"] for cat in json_struct["categories"]}
    label2id = {cat["name"]: cat["id"] for cat in json_struct["categories"]}
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
                "segmentation": [],
                "bbox": [],
                "category_id": [],
                "area": []
                }
        
        result[image_id]["bbox"].append(anot["bbox"]) # bbox: COCO format
        result[image_id]["segmentation"].append(anot["segmentation"]) # rle bin
        result[image_id]["category_id"].append(anot["category_id"])
        result[image_id]["area"].append(anot["area"])
    
    data = [result[k] for k in sorted(result.keys())]

    return width, height, id2label,label2id,images, Dataset.from_list(data)

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

def data_augmentation(train=False):
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    train_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
            # A.Resize(width=512, height=512),
            # A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ]
    )

    val_transform = A.Compose(
        [
            A.NoOp(),
            # A.Resize(width=512, height=512),
            # A.Normalize(mean=ADE_MEAN, std=ADE_STD),
        ]
    )
    if train:
        return train_transform
    else:
        return val_transform

def test(model, eval_dataloader, device):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_dataloader)):
            # Forward pass
            outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Loss
            loss = outputs.loss
            running_loss += loss.item()
            #print(f"AVG eval batch loss: {running_loss}")
            num_samples += 1

        print("Avg eval loss:", running_loss/num_samples)

def train(model, train_dataloader, eval_dataloader, lr=5e-5, weight_decay=1e-4, numb_epoch=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    times = []
    for epoch in range(numb_epoch):
        running_loss = 0.0
        num_samples = 0
        print("Epoch:", epoch)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            start_time = time.time()  # Start time
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()
            count = 0
            count_tot =0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    count_tot += 1
                    if param.grad.norm().item() < 1e-6:
                        count += 1
            print(f"Count grad minor 1e-6 {count}, total {count_tot}")

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()*batch_size
            num_samples += batch_size
            #print(f"AVG train batch loss: {loss.item()}")
            # Optimization
            optimizer.step()
            end_time = time.time()  # End time
            execution_time = (end_time - start_time)
            times.append(execution_time)

        print("AVG Train loss:", running_loss/ num_samples)
        break    
        #test(model, eval_dataloader, device)
    print("Time training:", np.mean(np.array(times)))

     
def debug1(dt):
    # print(dt[0].keys())
    d = dt[0]
    for k,v in d.items():
        if isinstance(v, torch.Tensor):
            print(k,v.shape)
        else:
            print(k,v)

    print("---")

def debug2(train_dataloader):
    batch = next(iter(train_dataloader))
    # for k,v in batch.items():
    #   if isinstance(v, torch.Tensor):
    #     print(k,v.shape)
    #   else:
    #     print(k,len(v))

    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

    batch_index = 1

    print("processed image shape: ", batch["pixel_values"][batch_index].shape)
    unnormalized_image = (batch["pixel_values"][batch_index].numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    print("processed tranposed image shape: ", unnormalized_image.shape)

    bgr_image = cv2.cvtColor(unnormalized_image, cv2.COLOR_RGB2BGR)

    # Save the image using OpenCV
    cv2.imwrite("unnormalized_image.png", bgr_image)

    print("----")
    print(batch["class_labels"][batch_index])
    print(id2label[x] for x  in batch["class_labels"][batch_index])
    print("mask shape", batch["mask_labels"][batch_index].shape)
    print("class labeñ shape", batch["class_labels"][batch_index].shape)

    print("id2label", id2label )
    print("Visualizing mask for:", id2label[batch["class_labels"][batch_index][0].item()])

    visual_mask = (batch["mask_labels"][batch_index][0].bool().numpy() * 255).astype(np.uint8)
    bgr_image = cv2.cvtColor(visual_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite("unnormalvisual_maskized_image.png", bgr_image)
    return batch

def debug3(model, batch):
     # save first loss
    outputs = model(
              pixel_values=batch["pixel_values"],
              mask_labels=batch["mask_labels"],
              class_labels=batch["class_labels"],
          )
    print(f"First output loss:{outputs.loss}")

def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)

  return visual_mask

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def combine_masks_with_colors(masks, colors):
    # Ensure all masks have the same shape (height, width)
    if len(set([mask.shape for mask in masks])) > 1:
        raise ValueError("All masks must have the same shape.")

    # Create an empty image (RGB) for the output
    combined_image = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

    # Loop through each mask and assign a unique color
    for i, mask in enumerate(masks):
        color = generate_random_color()  # Get a random color for each mask
        combined_image[mask > 0] = colors[i]  # Color the mask region

    return combined_image

def compute_metric(gt_json, pred_json):
    print("Final metric result: ")
    COCO_gt = COCO(gt_json)
    valid_image_ids = set(COCO_gt.getImgIds())
    print(f"Size preremoval: {len(pred_json)}")
    filtered_predictions = [
        pred for pred in pred_json if pred['image_id'] in valid_image_ids
    ]
    print(f"Size post removal: {len(filtered_predictions)}")
    coco_pred_filtered = COCO_gt.loadRes(filtered_predictions)

    coco_eval = COCOeval(COCO_gt, coco_pred_filtered, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_model(model, processor, dataset, device, eval_gt_path_json, output_json):
    first_image = True
    predictions_json = []
    for idx in range(len(dataset)):
        im_path = dataset[idx]["image_path"]
        frame_id = int ((im_path.split("/")[-1]).split(".")[0])
        video_id = int ((im_path.split("/")[-2])) * 10000
        image_id = frame_id + video_id
        image = np.array(Image.open(im_path).convert("RGB"))

        _,W,H = image.shape[::-1]

        inputs = processor(image, return_tensors="pt").to(device)
        inputs = inputs.to(device)
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(outputs, target_sizes=[(H,W)])[0]
        masks = []
        recovered_masks = []
        colors = []
        for segment in results['segments_info']:
            #print("Visualizing mask for instance:", model.config.id2label[segment['label_id']])
            mask = get_mask(results['segmentation'], segment['id'])
            mask[mask>0] = 1
            masks.append(mask)
            colors.append((255,0,0) if model.config.id2label[segment['label_id']] == "car" else (255,255,0) )
            rle = mask_utils.encode(np.asfortranarray(mask))
            #Convert bytes to a string for JSON serialization
            rle['counts'] = rle['counts'].decode('utf-8')
            prediction = {
                "image_id" : image_id,
                "score": segment["score"],
                "category_id": segment["label_id"],
                #"bbox": segment["boxes"][i].tolist(),
                "segmentation": rle,  # rle
            }
            predictions_json.append(prediction)
        
        if first_image:
            first_image = False
            cmask = combine_masks_with_colors(masks, colors)
            bgr_image = cv2.cvtColor(cmask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"first_image.png", bgr_image)

    with open(output_json, 'w') as json_file:
        json.dump(predictions_json, json_file, indent=4)
    
    compute_metric(eval_gt_path_json, predictions_json)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_output', required=False, default="inference_finetune_DETR.json", help='Output xml')
    parser.add_argument('--eval_gt_json_path', required=False, default="./eval_gt.json", help='Evaluation gt json path')
    parser.add_argument('--train_gt_json_path', required=False, default="./train_gt.json", help='Training gt json path')
    parser.add_argument('--save_model_name', required=False, default="./finetune_mask2former", help='Training gt json path')
    parser.add_argument('--predict', required=False, default="./finemask2form_pred.json", help='Training gt json path')
    args = parser.parse_args()
    print("tes")
    # alternatives? 
    checkpoint = "facebook/mask2former-swin-small-coco-instance"
    _,_,id2label,label2id,_,train_dataset = process_json(args.train_gt_json_path)
    _,_,_,_,_,eval_dataset = process_json(args.eval_gt_json_path)
    # print(train_dataset[0])
    # print(train_dataset[0]["segmentation"])
    print("evaldataset len",len(eval_dataset))
     
    data_aug = data_augmentation(train=True)
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    dt = ImageSegmentationDataset(train_dataset, processor, data_aug=data_aug)
    
    e_data_aug = data_augmentation(train=False)
    de = ImageSegmentationDataset(eval_dataset, processor, data_aug=e_data_aug)

    #debug1(dt)

    train_dataloader = DataLoader(dt, batch_size=4, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(de, batch_size=4, shuffle=False, collate_fn=collate_fn)

    batch = debug2(train_dataloader)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint,
                                                            id2label=id2label,
                                                            ignore_mismatched_sizes=True)
    
    debug3(model, batch)
   

    train(model, train_dataloader, eval_dataloader, numb_epoch=30, lr=5e-5)

    model.save_pretrained(args.save_model_name)
    processor.save_pretrained(args.save_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_model(model, processor, eval_dataset, device, args.eval_gt_json_path, args.predict)

     
    




    