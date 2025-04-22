import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
import torch.optim as optim
import torch
import pandas as pd
import evaluate
import tqdm
import sys
import os
import wandb
import time
import json
from transformers import ViTImageProcessor, AutoTokenizer, LlamaForCausalLM, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_IMG_START_TOKEN = "<IMG_START>"
DEFAULT_IMG_END_TOKEN = "<IMG_END>"

class Data(Dataset):
    def __init__(self, prefix, partition, feature_extractor, augment=False):
        self.prefix = prefix
        self.partition = partition
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if augment else None
    
    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        title, path = self.partition[idx]
        image_path = os.path.join(self.prefix, path)
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = self.transforms(image)
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values, title

class ViTLLamaModel(nn.Module):
    def __init__(self,
                 vit_model_name,
                 llama_model_name,
                 vit_pretrain,
                 freeze_vit=True,
                 img_start_token=DEFAULT_IMG_START_TOKEN,
                 img_end_token=DEFAULT_IMG_END_TOKEN,
                 prompt_text=None,
                 **kwargs
                 ):
        super(ViTLLamaModel, self).__init__()

        # --- ViT Encoder Setup ---
        # Using the encoder part directly from VisionEncoderDecoderModel might be fine,
        # but loading ViT standalone might offer more control if needed later.
        # Let's keep the original loading method for now.
        self.vit = VisionEncoderDecoderModel.from_pretrained(vit_model_name).encoder
        print(f"Loading ViT checkpoint from: {vit_pretrain}")
        checkpoint = torch.load(vit_pretrain, map_location=DEVICE)
        vit_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if "encoder" in k}
        self.vit.load_state_dict(vit_state_dict, strict=False)
        self.vit.to(DEVICE)

        if freeze_vit:
            print("Freezing ViT encoder.")
            for param in self.vit.parameters():
                param.requires_grad = False
        self.vit.eval()
        
        self.decoder = LlamaForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        
        # --- Add Special Tokens ---
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        special_tokens_dict = {'additional_special_tokens': [self.img_start_token, self.img_end_token]}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens: {self.img_start_token}, {self.img_end_token}")
        
        # Resize embeddings ONLY IF tokens were actually added
        if num_added_toks > 0:
             self.decoder.resize_token_embeddings(len(self.tokenizer))

        # Set pad token if necessary (Llama usually doesn't have one)
        if self.tokenizer.pad_token is None:
             print("Tokenizer has no pad token, setting it to eos_token.")
             self.tokenizer.pad_token = self.tokenizer.eos_token
             # Update Llama config if necessary (though usually handled internally)
             self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
             
        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.img_start_token)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.img_end_token)

        # --- Projection Layer (MLP) ---
        vit_hidden_size = self.vit.config.hidden_size
        llama_hidden_size = self.decoder.config.hidden_size
        # Use the Projector class defined earlier
        self.projection = nn.Sequential(
            nn.Linear(vit_hidden_size, llama_hidden_size),
            nn.GELU(),                              # Added GELU activation
            nn.Dropout(kwargs.get("projection_dropout",0.1)),
            nn.Linear(llama_hidden_size, llama_hidden_size),
            nn.GELU(),                              # Added GELU activation
            nn.Dropout(kwargs.get("projection_dropout",0.1)),
            nn.Linear(llama_hidden_size, llama_hidden_size) # Final output layer
        ).to(DEVICE)

        # --- Prompt Setup ---
        self.prompt_text = prompt_text if prompt_text else ""
        self.prompt_len = 0
        if self.prompt_text:
            print(f"Using prompt text: '{self.prompt_text}'")
            # Tokenize prompt - remove <s> but keep trailing space/quote if needed
            prompt_tokens_info = self.tokenizer(self.prompt_text, add_special_tokens=False, return_tensors="pt")
            self.prompt_ids = prompt_tokens_info.input_ids.to(DEVICE)
            self.prompt_len = self.prompt_ids.size(1) # Store prompt length
        else:
            print("No prompt text provided. Model will operate without prompt.")
            # Store empty tensor for consistency, though checks usually use prompt_len
            self.prompt_ids = torch.tensor([[]], dtype=torch.long, device=DEVICE)
            self.prompt_len = 0

    def get_input_embeddings(self):
        # Helper to access possibly PEFT-wrapped embeddings
        if hasattr(self.decoder, "get_input_embeddings"):
            return self.decoder.get_input_embeddings()
        elif hasattr(self.decoder, "model") and hasattr(self.decoder.model, "get_input_embeddings"): # Handle PEFT model structure
            return self.decoder.model.get_input_embeddings()
        else:
            # Fallback or error if structure is unexpected
            return self.decoder.embed_tokens # Common attribute name

    def forward(self, pixel_values, input_ids=None, labels=None, attention_mask_text=None):
        # pixel_values: (batch_size, num_channels, height, width)
        # labels: (batch_size, seq_len) - token IDs of the target text
        # 1. Get ViT Embeddings
        with torch.set_grad_enabled(not self.vit.training): # Use torch.no_grad() if ViT is in eval mode
             vit_outputs = self.vit(pixel_values=pixel_values)
        image_features = vit_outputs.last_hidden_state # (batch_size, vit_seq_len, vit_hidden_size)

        # 2. Project ViT Embeddings
        projected_features = self.projection(image_features) # (batch_size, vit_seq_len, llama_hidden_size)
        batch_size = projected_features.size(0)
        vit_seq_len = projected_features.size(1)

        # 3. Get Embeddings for Special Tokens and Text
        embedding_layer = self.get_input_embeddings()
        
        # Special token embeddings (batch_size, 1, llama_hidden_size)
        start_token_embeds = embedding_layer(torch.tensor([[self.img_start_token_id]] * batch_size, device=DEVICE))
        end_token_embeds = embedding_layer(torch.tensor([[self.img_end_token_id]] * batch_size, device=DEVICE))
        
        tensors_to_cat = [start_token_embeds, projected_features, end_token_embeds]
        prefix_len_no_text = 1 + vit_seq_len + 1 # Length before prompt/text
        if self.prompt_len > 0:
            prompt_embeds = embedding_layer(self.prompt_ids.expand(batch_size, -1))
            tensors_to_cat.append(prompt_embeds)
            prefix_len_no_text += self.prompt_len
        # Prompt embeddings (batch_size, prompt_len, llama_hidden_size)

        if input_ids is not None:
            # Target text label embeddings (batch_size, labels_len, llama_hidden_size)
            # Ensure labels don't contain pad tokens where embedding is needed (should be handled by collator)
            # If labels contain eos_token, include it for generation context
            
            label_embeds = embedding_layer(input_ids)
            tensors_to_cat.append(label_embeds)
            labels_len = input_ids.size(1)
            # 4. Concatenate Embeddings for Input
            # Order: [START | <ViT_FEATURES> | END | (<PROMPT_TEXT> - Optional) | <LABELS_TEXT>]
            
            inputs_embeds = torch.cat(tensors_to_cat, dim=1)

            # 5. Prepare Labels for Loss Calculation using the 'labels' tensor
            # The 'labels' tensor *already* has -100 for padding positions.
            # We just need to prepend -100 for the prefix tokens.
            adjusted_labels_for_loss = None
            ignore_prefix = torch.full((batch_size, prefix_len_no_text), -100, device=DEVICE, dtype=torch.long)
            adjusted_labels_for_loss = torch.cat([ignore_prefix, labels], dim=1) # Combine prefix ignore + label ignore

            # 6. Create Full Attention Mask by prepending 1s for the prefix
            attention_mask = None
            prefix_ones = torch.ones((batch_size, prefix_len_no_text), device=DEVICE, dtype=torch.long)
            attention_mask = torch.cat([prefix_ones, attention_mask_text], dim=1) # Combine prefix mask + text mask


            # 7. Forward pass through LLaMA Decoder
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                labels=adjusted_labels_for_loss, # Use the labels with -100 for prefix AND padding
                attention_mask=attention_mask      # Use the combined mask
            )
            return outputs # Contains loss and logits

        else:
            raise("Call model generation by the function generation(), dont use forward for inference!")


    # Keep generate method separate for clarity
    @torch.no_grad() # Ensure no gradients during generation
    def generate(self, pixel_values, max_new_tokens=50, **generate_kwargs):
         # pixel_values: (batch_size, num_channels, height, width)

        # 1. Get ViT Embeddings
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_features = vit_outputs.last_hidden_state

        # 2. Project ViT Embeddings
        projected_features = self.projection(image_features)
        batch_size = projected_features.size(0)

        # 3. Get Embeddings for Special Tokens and Prompt
        embedding_layer = self.get_input_embeddings()
        start_token_embeds = embedding_layer(torch.tensor([[self.img_start_token_id]] * batch_size, device=DEVICE))
        end_token_embeds = embedding_layer(torch.tensor([[self.img_end_token_id]] * batch_size, device=DEVICE))
        tensors_to_cat = [start_token_embeds, projected_features, end_token_embeds]

        # Conditionally add prompt embeddings based on initialization
        if self.prompt_len > 0:
            prompt_embeds = embedding_layer(self.prompt_ids.expand(batch_size, -1))
            tensors_to_cat.append(prompt_embeds)

        # 4. Concatenate Embeddings for Generation Prefix
        # Order: [START | <ViT_FEATURES> | END | <PROMPT_TEXT>]
        inputs_embeds = torch.cat(tensors_to_cat, dim=1)

        # 5. Create Attention Mask for the prefix
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=DEVICE, dtype=torch.long)

        # 7. Generate text using LLaMA's generate method
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs # Pass other generation params like temperature, top_k, etc.
        )

        return output_ids # Return token IDs

def optimizer_chooser(model, type_opt, config):
    if type_opt == "AdamW":
        return optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif type_opt == "Adam":
        return optim.Adam(model.parameters())
    elif type_opt == "SGD":
        return optim.SGD(model.parameters())
    else:
        print("Wrong optimizer type")
        sys.exit(1)

def apply_lora_to_decoder(model, lora_config):
    """Applies LoRA to the Llama decoder with broader target modules and saving embeddings."""
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        # Target modules similar to Code 2's example for broader adaptation
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
            ],
        lora_dropout=lora_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        # Ensure new embeddings (like special tokens) are saved
        modules_to_save=["embed_tokens"],
    )
    peft_model = get_peft_model(model, peft_config)
    print("Applied LoRA to LLaMA Decoder.")
    peft_model.print_trainable_parameters()
    return peft_model

def train(epochs, prefix, partitions, metric, config=None, llama_size="1b"):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_id}_llama_{llama_size}"
    wandb.init(project=config["wandb_project"], name=run_name, config=config)
    logging_dict = {}
    
    print(f"Training set size: {len(partitions['train'])}")
    print(f"Validation set size: {len(partitions['eval'])}")
    print(f"Test set size: {len(partitions['test'])}")

    print("\nSample Training Data (First 5):")
    sampled_train_images = []
    sampled_train_captions = []
    for i in range(min(5, len(partitions['train']))):
        title, path = partitions['train'][i]
        img = Image.open(os.path.join(prefix, path)).convert('RGB')
        print(f"Train Sample {i}: Title='{title}', Path='{path}'")
        sampled_train_images.append(img)
        sampled_train_captions.append(title)

    print("\nSample Validation Data (First 5):")
    sampled_val_images = []
    sampled_val_captions = []
    for i in range(min(5, len(partitions['eval']))):
        title, path = partitions['eval'][i]
        img = Image.open(os.path.join(prefix, path)).convert('RGB')
        print(f"Val Sample {i}: Title='{title}', Path='{path}'")
        sampled_val_images.append(img)
        sampled_val_captions.append(title)

    logging_dict.update({
        "train_samples": [wandb.Image(img, caption=title) for img, title in zip(sampled_train_images, sampled_train_captions)],
        "val_samples": [wandb.Image(img, caption=title) for img, title in zip(sampled_val_images, sampled_val_captions)],
    })

    vit_model_name = "nlpconnect/vit-gpt2-image-captioning"
    pretrained_model_path = config["pretrained_model_path"]
    llama_model_name = f"meta-llama/Llama-3.2-{llama_size.upper()}-Instruct"
    model = ViTLLamaModel(
        vit_model_name=vit_model_name,
        llama_model_name=llama_model_name,
        vit_pretrain=pretrained_model_path,
        freeze_vit=config["freeze_vit"],
        img_start_token=DEFAULT_IMG_START_TOKEN, # Pass special tokens
        img_end_token=DEFAULT_IMG_END_TOKEN,
        prompt_text=config["PROMPT"],
        projection_dropout=config["projection_dropout"]
    ).to(DEVICE)

    # Apply LoRA to the decoder
    lora_config = {
        "r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
    }
    model.decoder = apply_lora_to_decoder(model.decoder, lora_config)

    feature_extractor = ViTImageProcessor.from_pretrained(vit_model_name)
    tokenizer = model.tokenizer
    def collate_fn(batch):
        pixel_values = torch.stack([item[0] for item in batch])
        raw_texts = [item[1] for item in batch] # List of text strings
        raw_texts = [text + "." for text in raw_texts]

        # Tokenize texts, adding EOS token for generation context, and padding
        # Important: Add EOS token here, as LLaMA expects it for Causal LM task
        texts_with_eos = [text + tokenizer.eos_token for text in raw_texts]

        tokenized_outputs = tokenizer(
            texts_with_eos,
            padding="longest",        # Pad to longest in batch
            truncation=True,          # Truncate to max_length
            max_length=20,
            return_tensors="pt",
            return_attention_mask=True # *** Crucial: Get attention mask ***
        )

        # --- Prepare Tensors ---
        # 1. input_ids: Has actual pad_token_id. Used for embedding lookup.
        input_ids = tokenized_outputs.input_ids

        # 2. labels: Clone input_ids and replace pad_token_id with -100 for loss.
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # 3. attention_mask_text: Mask from tokenizer (1 for real tokens, 0 for padding).
        attention_mask_text = tokenized_outputs.attention_mask

        return pixel_values, input_ids, labels, attention_mask_text, raw_texts
    data_train = Data(prefix, partitions['train'], feature_extractor, augment=True)
    data_valid = Data(prefix, partitions['eval'], feature_extractor, augment=False)
    data_test = Data(prefix, partitions['test'], feature_extractor, augment=False)
    # Update DataLoaders to use the collate function
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8, collate_fn=collate_fn)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8, collate_fn=collate_fn)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    model.train()
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Manually check model params before training, number of trainable parameters is: {total_trainable_params:,}")
    optimizer = optimizer_chooser(model, config["optimizer_type"], config)
    crit = nn.CrossEntropyLoss(label_smoothing = config["label_smoothing"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    best_val_loss = float('inf')
    save_dir = os.path.join(config.get("save_dir", "./checkpoints"), run_name)
    os.makedirs(save_dir, exist_ok=True)
    patience = config["patience_es"]
    epochs_no_improve = 0
    warmup_epochs = config["warmup_ep"]
    initial_lr = 1e-5
    target_lr = config["lr"]
    min_lr = config.get("min_lr", 1e-7)

    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        if epoch < warmup_epochs:
            lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if epoch == warmup_epochs:
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
            # scheduler_cosine.step()

        train_loss = train_one_epoch(model, optimizer, crit, dataloader_train, tokenizer, config, epoch)
        print(f'train loss: {train_loss:.2f}, epoch: {epoch}')
        val_loss, val_metrics, val_wandb_dict = eval_epoch(model, crit, metric, dataloader_valid, tokenizer)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')

        # Added learning rate scheduling
        scheduler.step(val_loss)

        logging_dict.update({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": float(val_metrics.split("BLEU-1:")[1].split("%")[0]) / 100,
            "bleu2": float(val_metrics.split("BLEU2:")[1].split("%")[0]) / 100,
            "rouge_l": float(val_metrics.split("ROUGE-L:")[1].split("%")[0]) / 100,
            "meteor": float(val_metrics.split("METEOR:")[1].split("%")[0]) / 100,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        logging_dict.update(val_wandb_dict)
        wandb.log(logging_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config["save_cp"]:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    test_loss, test_metrics, test_wandb_dict = eval_epoch(model, crit, metric, dataloader_test, tokenizer)
    print(f'test loss: {test_loss:.2f}, metric: {test_metrics}')
    final_test_log = {"test_loss": test_loss}
    final_test_log["test_predictions"] = test_wandb_dict["eval_predictions"]
    final_test_log["test_ground_truths"] = test_wandb_dict["eval_ground_truths"]
    final_test_log["test_images"] = test_wandb_dict["eval_images"]

    wandb.log(final_test_log)
    wandb.finish()

def train_one_epoch(model, optimizer, crit, dataloader_train, tokenizer, config, epoch):
    total_loss = 0
    for batch in tqdm.tqdm(dataloader_train, desc="TRAINING MINI BATCH"):
        images, title_input_ids, title_for_loss, attn_mask_text, titles = batch
        images = images.to(DEVICE)
        title_input_ids = title_input_ids.to(DEVICE)
        title_for_loss = title_for_loss.to(DEVICE)
        attn_mask_text = attn_mask_text.to(DEVICE)
        
        # Removed the addition of "."
        # inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        outputs = model(
            pixel_values=images,
            input_ids=title_input_ids,
            labels=title_for_loss,
            attention_mask_text=attn_mask_text
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip_norm"])
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader_train)
    return avg_loss

def eval_epoch(model, crit, metric, dataloader, tokenizer):
    model.eval()
    preds, gts = [], []
    total_loss = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL MINI BATCH"):
            images, title_input_ids, title_for_loss, attn_mask_text, titles = batch
            images = images.to(DEVICE)
            title_input_ids = title_input_ids.to(DEVICE)
            title_for_loss = title_for_loss.to(DEVICE)
            attn_mask_text = attn_mask_text.to(DEVICE)
            # Removed the addition of "."
            # inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
            outputs = model(
                pixel_values=images,
                input_ids=title_input_ids,
                labels=title_for_loss,
                attention_mask_text=attn_mask_text
            )
            loss = outputs.loss
            total_loss += loss.item()
            total += 1
            output_ids = model.generate(pixel_values=images, 
                                        max_new_tokens=20, 
                                        num_beams=1,
                                        repetition_penalty=1.2,
                                        length_penalty=0.8,
                                        pad_token_id=tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # Removed the enforcement of closing "."
            filtered_preds = []
            for pred in predictions:
                if not pred.strip():
                    filtered_preds.append("[EMPTY]")
                else:
                    filtered_preds.append(pred)
            preds.extend(filtered_preds)
            gts.extend(titles)
    
    # Adjust metrics computation: strip the prompt but no closing "."
    prompt_length = len(config["PROMPT"])
    metric_preds = [pred[prompt_length:] for pred in preds]

    bleu, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=metric_preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=metric_preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=metric_preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=metric_preds, references=gts)['meteor']
    
    sample_indices = random.sample(range(len(preds)), min(9, len(preds)))
    sampled_preds = [preds[i] for i in sample_indices]
    sampled_gts = [gts[i] for i in sample_indices]
    sampled_images = [dataloader.dataset[i][0].cpu().numpy().transpose(1, 2, 0) for i in sample_indices]
    eval_wandb_dict = {
        "eval_predictions": sampled_preds,
        "eval_ground_truths": sampled_gts,
        "eval_images": [wandb.Image(img, caption=f"Pred: {pred}\nGT: {gt}") for img, pred, gt in zip(sampled_images, sampled_preds, sampled_gts)]
    }

    avg_eval_loss = total_loss / total if total > 0 else 0
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"
    return avg_eval_loss, result, eval_wandb_dict

if __name__ == "__main__":
    base_path = '/mnt/dataset/image_captioning_dataset/'
    
    with open('./DFDataSplit.json', 'r') as f:
        partitions = json.load(f)
    
    config = {
        "wandb_project": "DF_LORA_ViT_LLaMA",
        "prefix": "/mnt/dataset/image_captioning_dataset/FoodImages",
        "batch_size": 4,#8
        "optimizer_type": "AdamW",
        "lr": 5e-5,
        "min_lr": 1e-7,
        "label_smoothing": 0.1,
        "gradient_clip_norm": 2.0,
        "weight_decay": 0.05,
        "projection_dropout": 0.3,
        "num_epochs": 50,
        "warmup_ep": 1,
        "patience_es": 7,
        "save_dir": "./checkpoints",
        "pretrained_model_path": "./LORA/best_vitgpt2.pt",
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.3,
        "save_cp": False,
        "freeze_vit": True,
        "llama_size":"3b",
        "PROMPT" : ''
    }

    print(config)

    # partitions = np.load(splits_path, allow_pickle=True).item()
    # partitions['train'] = partitions['train'][:50]
    # partitions['eval'] = partitions['eval'][:50]
    # partitions['test'] = partitions['test'][:50]
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)
    train(config["num_epochs"], config["prefix"], partitions, metric, config=config, llama_size=config["llama_size"])