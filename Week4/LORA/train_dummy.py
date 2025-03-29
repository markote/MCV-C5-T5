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
from transformers import ViTImageProcessor, AutoTokenizer, LlamaForCausalLM, VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    def __init__(self, vit_model_name, llama_model_name, vit_pretrain, freeze_vit=True):
        super(ViTLLamaModel, self).__init__()
        # Load pre-trained ViT encoder
        self.vit = VisionEncoderDecoderModel.from_pretrained(vit_model_name).encoder
        print(vit_pretrain)
        checkpoint = torch.load(vit_pretrain, map_location=DEVICE)
        vit_state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items() if "encoder" in k}
        self.vit.load_state_dict(vit_state_dict, strict=False)
        self.vit.to(DEVICE)

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Load LLaMA decoder
        self.decoder = LlamaForCausalLM.from_pretrained(llama_model_name)
        # self.decoder.config.pad_token_id = AutoTokenizer.from_pretrained(llama_model_name).eos_token_id  # Set here
        
        # Projection layer to match ViT output to LLaMA input dimensions
        vit_hidden_size = self.vit.config.hidden_size  # Typically 768 for ViT
        llama_hidden_size = self.decoder.config.hidden_size  # 2048 for LLaMA 3.2-1B, 3072 for 3B
        self.projection = nn.Linear(vit_hidden_size, llama_hidden_size)

    def forward(self, pixel_values, labels=None, tokenizer=None):
        # Extract image features with frozen ViT
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_features = vit_outputs.last_hidden_state[:, 0, :]  # CLS token
        projected_features = self.projection(image_features)  # [batch_size, hidden_size]
        
        if labels is not None:
            # Tokenize inputs if not already tokenized (handled outside in this case)
            input_embeds = self.decoder.get_input_embeddings()(labels)  # [batch_size, seq_len, hidden_size]
            batch_size = projected_features.size(0)
            projected_features = projected_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
            decoder_inputs = torch.cat([projected_features, input_embeds], dim=1)  # [batch_size, 1 + seq_len, hidden_size]

            # Prepend a dummy token to labels to match sequence length
            dummy_token = torch.full((batch_size, 1), tokenizer.pad_token_id, device=DEVICE, dtype=labels.dtype)
            adjusted_labels = torch.cat([dummy_token, labels], dim=1)  # [batch_size, 1 + seq_len]

            # Pass to decoder with adjusted labels
            outputs = self.decoder(inputs_embeds=decoder_inputs, labels=adjusted_labels)
            return outputs
        else:
            # For generation
            projected_features = projected_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
            return self.decoder.generate(inputs_embeds=projected_features)

    def generate(self, pixel_values, **generate_kwargs):
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_features = vit_outputs.last_hidden_state[:, 0, :]
        projected_features = self.projection(image_features).unsqueeze(1)
        return self.decoder.generate(inputs_embeds=projected_features, **generate_kwargs)

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
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],  # Target attention layers in LLaMA
        lora_dropout=lora_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_config)

def train(epochs, prefix, partitions, metric, config=None, llama_size="1b"):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_id}_llama_{llama_size}"
    wandb.init(project="image_captioning_LORA_ViT_LLaMA", name=run_name, config=config)
    logging_dict = {}
    
    # Dataset inspection
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

    # Model setup
    vit_model_name = "nlpconnect/vit-gpt2-image-captioning"
    pretrained_model_path = config["pretrained_model_path"]  # Path to your saved checkpoint
    llama_model_name = f"meta-llama/Llama-3.2-{llama_size.upper()}-Instruct"  # "1B" or "3B"
    model = ViTLLamaModel(vit_model_name, llama_model_name, vit_pretrain = pretrained_model_path).to(DEVICE)

    # Apply LoRA to the decoder
    lora_config = {
        "r": config["lora_r"],  # Rank of the low-rank matrices
        "lora_alpha": config["lora_alpha"],  # Scaling factor
        "lora_dropout": config["lora_dropout"],
    }
    model.decoder = apply_lora_to_decoder(model.decoder, lora_config)

    # Feature extractor and tokenizer
    feature_extractor = ViTImageProcessor.from_pretrained(vit_model_name)
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA uses EOS as pad token

    # DataLoaders
    data_train = Data(prefix, partitions['train'], feature_extractor, augment=True)
    data_valid = Data(prefix, partitions['eval'], feature_extractor, augment=False)
    data_test = Data(prefix, partitions['test'], feature_extractor, augment=False)
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    
    # Training setup
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"], config)
    crit = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"], ignore_index=tokenizer.pad_token_id)

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
    for images, titles in dataloader_train:
        images = images.to(DEVICE)
        inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        outputs = model(pixel_values=images, labels=inputs, tokenizer = tokenizer)
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
        for images, titles in dataloader:
            images = images.to(DEVICE)
            inputs = tokenizer(titles, padding=True, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
            outputs = model(pixel_values=images, labels=inputs, tokenizer = tokenizer)
            loss = outputs.loss
            total_loss += loss.item()
            total += 1
            output_ids = model.generate(pixel_values=images, max_length=80, num_beams=4, pad_token_id=tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds.extend(predictions)
            gts.extend(titles)
    
    bleu, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']
    
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
    splits_path = f'FilteredDataSplit.npy'

    config = {
        "prefix": "/mnt/dataset/image_captioning_dataset/FoodImages",
        "batch_size": 16,
        "optimizer_type": "AdamW",
        "lr": 1e-5,
        "min_lr": 1e-7,
        "label_smoothing": 0.1,
        "gradient_clip_norm": 2.0,
        "dropout_enc": 0.1,
        "dropout_dec": 0.1,
        "weight_decay": 0.01,
        "num_epochs": 30,
        "warmup_ep": 1,
        "patience_es": 7,
        "save_dir": "./checkpoints",
        "pretrained_model_path":"./LORA/best_vitgpt2.pt",
        "lora_r": 16,  # Rank of the low-rank matrices
        "lora_alpha": 32,  # Scaling factor
        "lora_dropout": 0.1
    }

    print(config)

    partitions = np.load(splits_path, allow_pickle=True).item()
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)

    # Train with LLaMA 3.2-1B
    print("Training with LLaMA 3.2-1B")
    train(config["num_epochs"], config["prefix"], partitions, metric, config=config, llama_size="1b")

    # Train with LLaMA 3.2-3B (uncomment to run)
    # print("Training with LLaMA 3.2-3B")
    # train(config["num_epochs"], config["prefix"], partitions, metric, config=config, llama_size="3b")