import os
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import json
from tqdm import tqdm
import evaluate
import numpy as np
from PIL import Image
import random
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class Data(torch.utils.data.Dataset):
    def __init__(self, prefix, partition, data_aug=False):
        self.prefix = prefix
        self.partition = partition

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        title, path = self.partition[idx]
        
        # Return image path and title instead of tensor
        image_path = os.path.join(self.prefix, path)
        
        return image_path, title


# Prediction step function (modified as per your request)
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


# Evaluation function
def eval_epoch(model, metric, dataloader):
    model.eval()
    gts = []
    preds = []
    all_images = []
    
    with torch.no_grad():
        for i, (image_paths, titles) in enumerate(dataloader):
            # Predict captions using the modified predict_step function
            pred = predict_step(image_paths)

            # Store ground truth and predicted captions
            gts.extend(titles)
            preds.extend(pred)

            # Print every 5th pair of GT and Pred
            if (i + 1) % 1 == 0:
                print(f"GT: {titles[0]}")
                print(f"Pred: {pred[0]}")
                print("----")

    bleu, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']

    # Logging to wandb
    if len(preds) >= 9:
        sample_indices = random.sample(range(len(preds)), 9)
        sampled_preds = [preds[i] for i in sample_indices]
        sampled_gts = [gts[i] for i in sample_indices]
        print("Eval preds (9 random): ", sampled_preds)
        print("Eval gts (9 random): ", sampled_gts)
    else:
        print("Eval preds: ", preds)
        print("Eval gts: ", gts)

    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU-2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"
    return result


# Run evaluation

# Load ViT-GPT Model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

base_path = '/ghome/c5mcv05/image_captioning_dataset/'
img_path = f'{base_path}FoodImages/'
splits_path = f'{base_path}FilteredDataSplitGoodroot.npy'
prefix= "/ghome/c5mcv05/image_captioning_dataset/FoodImages"

max_length = 50
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

partitions = np.load(splits_path, allow_pickle=True).item()

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
metric = (bleu, rouge, meteor)

data_train = Data(prefix, partitions['train'], data_aug=True)
data_valid = Data(prefix, partitions['eval'])
data_test = Data(prefix, partitions['test'])
dataloader_train = DataLoader(data_train, 32, pin_memory=True, shuffle=True, num_workers=8)
dataloader_valid = DataLoader(data_valid, 32, pin_memory=True, shuffle=False, num_workers=8)
dataloader_test = DataLoader(data_test, 32, pin_memory=True, shuffle=False, num_workers=8)

test_metrics = eval_epoch(model, metric, dataloader_valid)
print(f"Metrics: {test_metrics}")
