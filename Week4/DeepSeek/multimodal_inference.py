from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
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

class Data(Dataset):
    def __init__(self, prefix, partition, data_aug=False):
        self.prefix = prefix
        self.partition = partition
        self.max_len = TEXT_MAX_LEN
        if data_aug:
            self.img_proc = torch.nn.Sequential(
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=10),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )
        else:
            self.img_proc = torch.nn.Sequential(
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224), antialias=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        title, path = self.partition[idx]
        # Image processing
        path = os.path.basename(path)
        # img = Image.open(os.path.join(self.prefix, path)).convert('RGB')
        # img = self.img_proc(img)
    
        ## caption processing
        # print("Image captioning processing: ")
        # print(title)
        # words = ["<SOS>"]
        # words.extend(nltk.word_tokenize(title)) # vector of words we need to add <EOS> and <PAD>
        # words.extend(["<EOS>"])
        # gap = self.max_len - len(words)
        # words.extend(["<PAD>"]*gap)
        # cap_idx = [TOKEN2IDX[i] for i in words]
        # print("final list to idx", final_list)
        # print("final idx", cap_idx)
        # print("final idx in pytorch tensor: ",  torch.tensor(cap_idx, dtype=torch.long))
        # sys.exit(1)
        
        return title, path

def generation_conversation(path_image, prompt):
    return [
            {
                "role": "<|User|>",
                "content": f"<image>\n<|ref|>{prompt}<|/ref|>.",
                "images": [path_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]


def inference_one_split(dataset, metric, model=None, prompt="Say 'error wrong prompt for this result' ignoring the given image", split=""):
   
    for path_image, title in dataset:
        ## single image conversation example
        conversation = generation_conversation(path_image, prompt)

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        break

def inference(prefix, partitions, metric, config=None, run_name=""):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{run_id}"
    wandb.init(project="Inference multimodal deepseek model", name=run_name, config=config)
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl2"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    data_train = Data(prefix, partitions['train'])
    data_valid = Data(prefix, partitions['eval'])
    data_test = Data(prefix, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    inference_one_split(data_train, metric, model=vl_gpt, promp=config["prompt"], split="train")
    inference_one_split(data_valid, metric, model=vl_gpt, promp=config["prompt"], split="valid")
    inference_one_split(data_test, metric, model=vl_gpt, promp=config["prompt"], split="test")

    wandb.finish()


if __name__ == "__main__":
    base_path = '/ghome/c5mcv05/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}FilteredDataSplit.npy'

    config = {
        "prefix": "/ghome/c5mcv05/image_captioning_dataset/FoodImages/",
        "batch_size": 32,
        "prompt":"This image comes from a web of cooking recipe, guess the title of the recipe based on the image.",
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

