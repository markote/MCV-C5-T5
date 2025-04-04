# from transformers import AutoModelForCausalLM
# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from deepseek_vl2.utils.io import load_pil_images
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import evaluate
import tqdm
import torch
import sys
import os
import wandb
import time
import pickle
import json
import random
from PIL import Image

class Data(Dataset):
    def __init__(self, prefix, partition):
        self.prefix = prefix
        self.partition = partition

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        title, path = self.partition[idx]
        path = os.path.join(self.prefix, os.path.basename(path))
        
        return title, path

def generation_conversation(path_image, prompt, resize_size=224):
    return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": path_image,
                            "resized_height": resize_size,
                            "resized_width": resize_size,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

def avg(list_param):
    return sum(list_param)/len(list_param)

def inference_one_split(dataset, metric, model=None, processor=None, prompt="Say 'error wrong prompt for this result' ignoring the given image", split=""):
    predictions = []
    gts = []
    all_images = []
    bleux1, bleux2, rouges, meteores = [],[],[],[]
    bleu, rouge, meteor = metric
    for title, path_image in dataset:
        conversation = generation_conversation(path_image, prompt)
        gts.append([title])
        all_images.append(path_image)
        # print("Title:", title)
        # print("Image path:", path_image)
        # print("Conversation:", conversation)

        # Preparation for inference
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print("Output text:", output_text[0])
        predictions.append(output_text[0])
        if len(gts) > 5000:
            bleu1 = bleu.compute(predictions=predictions, references=gts, max_order=1)["bleu"]
            bleu2 = bleu.compute(predictions=predictions, references=gts, max_order=2)["bleu"]
            res_r = rouge.compute(predictions=predictions, references=gts)['rougeL']
            res_m = meteor.compute(predictions=predictions, references=gts)['meteor']
            bleux1.append(bleu1)
            bleux2.append(bleu2)
            rouges.append(res_r)
            meteores.append(res_m)
            wandb.log({
                f"{split}_bleu1": bleu1,
                f"{split}_bleu2": bleu2,
                f"{split}_rouge_l": res_r,
                f"{split}_meteor": res_m
            })
            sample_indices = random.sample(range(len(predictions)), 9)
            sampled_preds = [predictions[i] for i in sample_indices]
            sampled_gts = [gts[i] for i in sample_indices]
            sampled_images = [all_images[i] for i in sample_indices]
            print("Eval preds (9 random): ", sampled_preds)
            print("Eval gts (9 random): ", sampled_gts)
            wandb.log({
                f"{split}_predictions": sampled_preds,
                f"{split}_ground_truths": sampled_gts,
                f"{split}_images": [wandb.Image(Image.open(img_path), caption=f"Pred: {pred}\nGT: {gt[0]}") for img_path, pred, gt in zip(sampled_images, sampled_preds, sampled_gts)]
            })
            predictions = []
            gts = []
            print("5k done.")
            
            
    # print("Predictions:", predictions)
    # print("GTs :", gts)
    bleu1 = bleu.compute(predictions=predictions, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=predictions, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=predictions, references=gts)['rougeL']
    res_m = meteor.compute(predictions=predictions, references=gts)['meteor']
    wandb.log({
            f"{split}_bleu1": bleu1,
            f"{split}_bleu2": bleu2,
            f"{split}_rouge_l": res_r,
            f"{split}_meteor": res_m
        })
    bleux1.append(bleu1)
    bleux2.append(bleu2)
    rouges.append(res_r)
    meteores.append(res_m)
    wandb.log({
            f"AVG_{split}_bleu1": avg(bleux1),
            f"AVG_{split}_bleu2": avg(bleux2),
            f"AVG_{split}_rouge_l": avg(rouges),
            f"AVG_{split}_meteor": avg(meteores)
        })
   

    sample_indices = random.sample(range(len(predictions)), 9)
    sampled_preds = [predictions[i] for i in sample_indices]
    sampled_gts = [gts[i] for i in sample_indices]
    sampled_images = [all_images[i] for i in sample_indices]
    print("Eval preds (9 random): ", sampled_preds)
    print("Eval gts (9 random): ", sampled_gts)
    wandb.log({
        f"{split}_predictions": sampled_preds,
        f"{split}_ground_truths": sampled_gts,
        f"{split}_images": [wandb.Image(Image.open(img_path), caption=f"Pred: {pred}\nGT: {gt[0]}") for img_path, pred, gt in zip(sampled_images, sampled_preds, sampled_gts)]
    })

def inference(prefix, partitions, metric, config=None, run_name=""):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{run_id}"
    wandb.init(project="Inference multimodal qwen model", name=run_name, config=config)
    
    checkpoint = "Qwen/Qwen2.5-VL-3B-Instruct"
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    # default processer
    processor = AutoProcessor.from_pretrained(checkpoint)

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    #data_train = Data(prefix, partitions['train'])
    data_valid = Data(prefix, partitions['eval'])
    data_test = Data(prefix, partitions['test'])
    inference_one_split(data_train, metric, model=model, processor=processor, prompt=config["prompt"], split="train")
    inference_one_split(data_valid, metric, model=model, processor=processor, prompt=config["prompt"], split="valid")
    inference_one_split(data_test, metric, model=model, processor=processor, prompt=config["prompt"], split="test")

    wandb.finish()


if __name__ == "__main__":
    base_path = '/ghome/c5mcv05/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}FilteredDataSplit.npy'

    config = {
        "prefix": "/mnt/dataset/image_captioning_dataset/FoodImages/",
        "batch_size": 32,
        "prompt":"This image comes from a web of cooking recipes, guess the title of the recipe based on the image. Only output your guessed title.",
    }

    partitions = None
    with open('./FilteredDataSplit.json', 'r') as f:
        partitions = json.load(f)

    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)
    print("Prefix: ", config["prefix"])
    run_name=f'Inference multimodal Language Model'
    inference(config["prefix"], partitions, metric, config=config, run_name=run_name)

