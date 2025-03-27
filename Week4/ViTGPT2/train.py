import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, ViTConfig, GPT2Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Data(Dataset):
    def __init__(self, prefix, partition, feature_extractor):
        self.prefix = prefix
        self.partition = partition
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.partition)

    def __getitem__(self, idx):
        title, path = self.partition[idx]
        image_path = os.path.join(self.prefix, path)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return pixel_values, title




def optimizer_chooser(model, type_opt, config):
    if type_opt == "AdamW":
        return optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif type_opt == "Adam":
        return optim.Adam(model.parameters())
    elif type_opt == "SGD":
        return optim.SGD(model.parameters())
    else:
        print("Wrong model")
        sys.exit(1)


def train(epochs, prefix, partitions, metric, config=None):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_id}"
    wandb.init(project="image_captioning_VITGPT2", name=run_name, config=config)
    logging_dict = {}
    
    # Added dataset inspection before training
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

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
    dropout_enc = config.get("dropout_enc", 0.0)
    dropout_dec = config.get("dropout_dec", 0.0)


    # Update dropout in the ViT encoder layers
    for layer in model.encoder.encoder.layer:
        layer.attention.attention.dropout.p = dropout_enc  # Set dropout for self-attention
        layer.attention.output.dropout.p = dropout_enc  # Set dropout for attention output
        layer.output.dropout.p = dropout_enc  # Set dropout for feedforward output

    # Update dropout in the ViT embeddings layer (before the encoder)
    model.encoder.embeddings.dropout.p = dropout_enc  # Set dropout for embeddings

    # Update dropout in the GPT2 decoder layers
    for block in model.decoder.transformer.h:
        block.attn.attn_dropout.p = dropout_dec  # Attention dropout
        block.attn.resid_dropout.p = dropout_dec  # Residual dropout
        block.mlp.dropout.p = dropout_dec  # MLP dropout
        block.crossattention.attn_dropout.p = dropout_dec  # Cross-attention dropout
        block.crossattention.resid_dropout.p = dropout_dec  # Cross-attention residual dropout

    print(model)

    # Initialize the feature extractor and tokenizer
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Enable data augmentation for training
    data_train = Data(prefix, partitions['train'],feature_extractor)
    data_valid = Data(prefix, partitions['eval'],feature_extractor)
    data_test = Data(prefix, partitions['test'],feature_extractor)
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    
    
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"], config)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none', ignore_index=2)

    # Added learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    base_save_dir = config.get("save_dir", "./checkpoints")
    save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    patience = config["patience_es"]
    epochs_no_improve = 0
    warmup_epochs = config["warmup_ep"]  # Warm up for x epochs
    initial_lr = 1e-5  # Start with a small learning rate
    target_lr = config["lr"]
    min_lr = config.get("min_lr", 1e-7)  # Minimum learning rate for cosine decay
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    SWITCH = False


    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler_cosine.step()  # Apply cosine decay


        train_loss = train_one_epoch(model, optimizer, crit, dataloader_train,tokenizer,config,epoch,SWITCH)
        print(f'train loss: {train_loss:.2f}, epoch: {epoch}')
        val_loss, val_metrics, val_wandb_dict = eval_epoch(model, crit, metric, dataloader_valid,tokenizer)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')

        # Added learning rate scheduling
        # scheduler.step(val_loss)

        logging_dict.update({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": float(val_metrics.split("BLEU-1:")[1].split("%")[0]) / 100,
            "bleu2": float(val_metrics.split("BLEU2:")[1].split("%")[0]) / 100,
            "rouge_l": float(val_metrics.split("ROUGE-L:")[1].split("%")[0]) / 100,
            "meteor": float(val_metrics.split("METEOR:")[1].split("%")[0]) / 100,
            "learning_rate": optimizer.param_groups[0]['lr'],  # Added logging
        })
        logging_dict.update(val_wandb_dict)
        wandb.log(logging_dict)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    test_loss, test_metrics, test_wandb_dict = eval_epoch(model, crit, metric, dataloader_test)
    print(f'test loss: {test_loss:.2f}, metric: {test_metrics}')
    final_test_log = {"test_loss": test_loss}
    final_test_log["test_predictions"] = test_wandb_dict["eval_predictions"]
    final_test_log["test_ground_truths"] = test_wandb_dict["eval_ground_truths"]
    final_test_log["test_images"] = test_wandb_dict["eval_images"]

    wandb.log(test_wandb_dict)

    wandb.finish()


def train_one_epoch(model, optimizer, crit, dataloader_train,tokenizer, config,epoch,SWITCH):
    total_loss = 0

    if config["train_mode"] == "encoder":
        # Freeze decoder layers (train only the encoder)
        for param in model.encoder.parameters():
            param.requires_grad = True  # Train encoder
        for param in model.decoder.parameters():
            param.requires_grad = False  # Freeze decoder

    elif config["train_mode"] == "decoder":
        # Freeze encoder layers (train only the decoder)
        for param in model.encoder.parameters():
            param.requires_grad = False  # Freeze encoder
        for param in model.decoder.parameters():
            param.requires_grad = True  # Train decoder

    if config["train_mode"] == "alternate" and epoch % config["switch_epochs"] == 0:
        if SWITCH:
            # Freeze decoder and train encoder
            print("Encoder unfrozen, decoder frozen")
            SWITCH = not SWITCH
            for param in model.encoder.parameters():
                param.requires_grad = True  # Train encoder
            for param in model.decoder.parameters():
                param.requires_grad = False  # Freeze decoder
        else:
            # Freeze encoder and train decoder
            print("Encoder frozen, decoder unfrozen")
            SWITCH = not SWITCH
            for param in model.encoder.parameters():
                param.requires_grad = False  # Freeze encoder
            for param in model.decoder.parameters():
                param.requires_grad = True  # Train decoder
    
    for images, titles in dataloader_train:
        images = images.to(DEVICE)
        inputs = tokenizer(titles, padding=True, return_tensors="pt").input_ids.to(DEVICE)
        outputs = model(pixel_values=images, labels=inputs)
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
            inputs = tokenizer(titles, padding=True, return_tensors="pt").input_ids.to(DEVICE)

            # Forward pass
            outputs = model(pixel_values=images, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()
            total += 1
            
            # Generate predictions
            output_ids = model.generate(images, max_length=80, num_beams=4)
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds.extend(predictions)
            gts.extend(titles)
    
    # Compute metrics
    bleu, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']
    
    # Randomly select 9 samples for logging
    sample_indices = random.sample(range(len(preds)), min(9, len(preds)))
    sampled_preds = [preds[i] for i in sample_indices]
    sampled_gts = [gts[i] for i in sample_indices]
    sampled_images = [dataloader.dataset[i][0] for i in sample_indices]  # Load images from dataset

    # Convert tensors to numpy for logging
    sampled_images_np = [img.cpu().numpy().transpose(1, 2, 0) for img in sampled_images]

    # Log predictions to W&B
    eval_wandb_dict = {
        "eval_predictions": sampled_preds,
        "eval_ground_truths": sampled_gts,
        "eval_images": [wandb.Image(img, caption=f"Pred: {pred}\nGT: {gt}") for img, pred, gt in zip(sampled_images_np, sampled_preds, sampled_gts)]
    }

    avg_eval_loss = total_loss / total if total > 0 else 0
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"
    
    return avg_eval_loss, result, eval_wandb_dict

if __name__ == "__main__":
    base_path = '/ghome/c5mcv05/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}FilteredDataSplitGoodroot.npy'

    config = {
        "prefix": "/ghome/c5mcv05/image_captioning_dataset/FoodImages",
        "testdata_path": "~/datanew/MIT_small_train_2/test",
        "train_mode": "decoder",  # "alternate", "encoder", "decoder" options
        "switch_epochs": 3, # on how many epochs alternate freezing enc or dec
        "batch_size": 32,
        "optimizer_type": "AdamW",
        "lr": 1e-5,
        "min_lr": 1e-7,                      # Minimum learning rate for cosine decay
        "label_smoothing": 0.1,              # Label smoothing to improve generalization
        "gradient_clip_norm": 2.0,           # Gradient clipping threshold
        "dropout_enc": 0.5,
        "dropout_dec": 0.5,
        "weight_decay": 0.01,
        "num_epochs": 20,
        "warmup_ep": 1,
        "patience_es": 15,
        "save_dir": "./checkpoints",
    }

    print(config)

    partitions = np.load(splits_path, allow_pickle=True).item()
    
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)

    train(config["num_epochs"], config["prefix"], partitions, metric, config=config)