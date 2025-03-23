import numpy as np
import random
from transformers import ResNetModel, AutoTokenizer
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

# Load the RoBERTa tokenizer for subword (BPE) tokenization
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
VOCAB_SIZE = tokenizer.vocab_size  # 50,265 for roberta-base
TEXT_MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, prefix, partition, data_aug=False):
        self.prefix = prefix
        self.partition = partition
        self.max_len = TEXT_MAX_LEN
        self.tokenizer = tokenizer
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
        img = Image.open(os.path.join(self.prefix, path)).convert('RGB')
        img = self.img_proc(img)
    
        # Caption processing with subword (BPE) tokenization
        encoded = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        
        return img, input_ids.to(dtype=torch.long)


class Model(nn.Module):
    def __init__(self, encoder_type='resnet18', decoder_type='gru', apply_teacher_forcing=False):
        super().__init__()
        if encoder_type == "resnet18":
            print("resnet18 encoder")
            self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        elif encoder_type == 'resnet34':
            self.resnet = ResNetModel.from_pretrained('microsoft/resnet-34').to(DEVICE)
        else:
            raise ValueError("Unsupported encoder. Choose 'resnet18' or 'resnet34'.")

        if decoder_type == "gru":
            print("gru decoder")
            self.decoder = nn.GRU(512, 512, num_layers=1, dropout=0.5)
            self.zero_cell = torch.zeros(1, 512, device=DEVICE)
        elif decoder_type == 'lstm':
            print("lstm decoder")
            self.decoder = nn.LSTM(512, 512, num_layers=1, dropout=0.5)
            self.zero_cell = torch.zeros(1, 512, device=DEVICE)
        else:
            raise ValueError("Unsupported decoder. Choose 'gru' or 'lstm'.")
           
        self.apply_teacher_forcing = apply_teacher_forcing
        self.teacher_forcing_ratio = 1.0 if apply_teacher_forcing else 0.0  # Added for scheduled sampling
        self.proj = nn.Linear(512, VOCAB_SIZE)
        self.embed = nn.Embedding(VOCAB_SIZE, 512)
        self.layer_norm = nn.LayerNorm(512).to(DEVICE)
        self.start = torch.tensor(tokenizer.cls_token_id, device=DEVICE)

    # Added method to set teacher forcing ratio for scheduled sampling
    def set_teacher_forcing_ratio(self, ratio):
        self.teacher_forcing_ratio = ratio

    def forward(self, img, titles=None):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)

        if titles is not None and self.training and self.apply_teacher_forcing and random.random() < self.teacher_forcing_ratio:  # Modified for scheduled sampling
            embeds = self.embed(titles[:, :-1])
            embeds = embeds.permute(1, 0, 2)
            if isinstance(self.decoder, nn.LSTM):
                out, _ = self.decoder(embeds, (feat, self.zero_cell.repeat(1, batch_size, 1)))
            else:
                out, _ = self.decoder(embeds, feat)
            out = self.layer_norm(out)
            res = self.proj(out.permute(1, 0, 2))
            return res.permute(0, 2, 1)
        else:
            start_embed = self.embed(self.start).repeat(batch_size, 1).unsqueeze(0)
            inp = start_embed
            hidden = feat
            outputs = []
            for t in range(TEXT_MAX_LEN - 1):
                if isinstance(self.decoder, nn.LSTM):
                    out, (hidden, _) = self.decoder(inp, (hidden, self.zero_cell.repeat(1, batch_size, 1)))
                else:
                    out, hidden = self.decoder(inp, hidden)
                out = self.layer_norm(out)
                out = self.proj(out.permute(1, 0, 2)).permute(0, 2, 1)
                outputs.append(out)
                _, predicted = out.max(1)
                inp = self.embed(predicted).permute(1, 0, 2)
            res = torch.cat(outputs, dim=2)
            return res


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
    wandb.init(project="image_captioning", name=run_name, config=config)
    
    encoder_type = config.get("encoder_type", "resnet18")
    decoder_type = config.get("decoder_type", "gru")
    
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

    wandb.log({
        "train_samples": [wandb.Image(img, caption=title) for img, title in zip(sampled_train_images, sampled_train_captions)],
        "val_samples": [wandb.Image(img, caption=title) for img, title in zip(sampled_val_images, sampled_val_captions)],
    })

    # Enable data augmentation for training
    data_train = Data(prefix, partitions['train'], data_aug=True)
    data_valid = Data(prefix, partitions['eval'])
    data_test = Data(prefix, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    model = Model(encoder_type=encoder_type, decoder_type=decoder_type, apply_teacher_forcing=config["apply_teacher_forcing"]).to(DEVICE)
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"], config)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none', ignore_index=tokenizer.pad_token_id)

    # Added learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    base_save_dir = config.get("save_dir", "./checkpoints")
    save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    patience = config["patience_es"]
    epochs_no_improve = 0
    warmup_epochs = config["warmup_ep"]  # Warm up for x epochs
    initial_lr = 1e-5  # Start with a small learning rate
    target_lr = config["lr"] 

    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # Added scheduled sampling
        if config["apply_teacher_forcing"]:
            teacher_forcing_ratio = max(0.0, 1.0 - (epoch / config["schedule_sampling_speed"]))
            model.set_teacher_forcing_ratio(teacher_forcing_ratio)
            print(f"Epoch {epoch}, Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")

        train_loss = train_one_epoch(model, optimizer, crit, dataloader_train, accum_steps=config.get("accum_steps", 4), apply_teacher_forcing=config["apply_teacher_forcing"])
        print(f'train loss: {train_loss:.2f}, epoch: {epoch}')
        val_loss, val_metrics = eval_epoch(model, crit, metric, dataloader_valid)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')

        # Added learning rate scheduling
        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": float(val_metrics.split("BLEU-1:")[1].split("%")[0]) / 100,
            "bleu2": float(val_metrics.split("BLEU2:")[1].split("%")[0]) / 100,
            "rouge_l": float(val_metrics.split("ROUGE-L:")[1].split("%")[0]) / 100,
            "meteor": float(val_metrics.split("METEOR:")[1].split("%")[0]) / 100,
            "teacher_forcing_ratio": teacher_forcing_ratio if config["apply_teacher_forcing"] else 0.0,  # Added logging
            "learning_rate": optimizer.param_groups[0]['lr'],  # Added logging
        })

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

    test_loss, test_metrics = eval_epoch(model, crit, metric, dataloader_test)
    print(f'test loss: {test_loss:.2f}, metric: {test_metrics}')
    wandb.log({"test_loss": test_loss})

    wandb.finish()


def train_one_epoch(model, optimizer, crit, dataloader, accum_steps=4, apply_teacher_forcing=False):
    model.train()    
    train_loss = 0.0
    total = 0
    batch_losses = []  # Added for detailed logging
    optimizer.zero_grad()
    
    for i, (images, titles) in enumerate(dataloader):
        images, titles = images.to(DEVICE), titles.to(DEVICE)
        
        outputs = model(images, titles)
        
        if apply_teacher_forcing:
            loss = crit(outputs, titles[:, 1:])
            loss = loss.mean() / accum_steps
        else:
            batch_size, _, seq_len = outputs.shape
            loss = crit(outputs, titles[:, 1:])
            mask = torch.ones_like(loss, device=DEVICE)
            for b in range(batch_size):
                gt_eos_pos = (titles[b, 1:] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(gt_eos_pos) > 0:
                    gt_eos_pos = gt_eos_pos[0].item()
                else:
                    gt_eos_pos = seq_len
                _, predicted = outputs[b].max(0)
                pred_eos_pos = (predicted == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(pred_eos_pos) > 0:
                    pred_eos_pos = pred_eos_pos[0].item()
                else:
                    pred_eos_pos = seq_len
                eos_pos = min(gt_eos_pos, pred_eos_pos) + 1
                mask[b, eos_pos:] = 0
            loss = (loss * mask).sum() / (mask.sum() + 1e-8) / accum_steps
        
        loss.backward()
        # Added gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Added logging for batch loss
        batch_loss = loss.item() * accum_steps
        batch_losses.append(batch_loss)
        print(f"Batch {i}, Train Batch Loss: {batch_loss:.4f}")
        train_loss += batch_loss * images.size(0)
        total += images.size(0)
    
    avg_train_loss = train_loss / total
    print(f"Average Train Loss for Epoch: {avg_train_loss:.4f}")
    return avg_train_loss


def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    eval_loss = 0.0
    total = 0
    gts = []
    preds = []
    all_images = []
    batch_losses = []  # Added for detailed logging
    with torch.no_grad():
        for i, (images, titles) in enumerate(dataloader):
            images, titles = images.to(DEVICE), titles.to(DEVICE)

            outputs = model(images)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: NaN or Inf in outputs")
            
            batch_size, _, seq_len = outputs.shape
            loss = crit(outputs, titles[:, 1:])
            mask = torch.ones_like(loss, device=DEVICE)
            for b in range(batch_size):
                gt_eos_pos = (titles[b, 1:] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(gt_eos_pos) > 0:
                    gt_eos_pos = gt_eos_pos[0].item()
                else:
                    gt_eos_pos = seq_len
                _, predicted = outputs[b].max(0)
                pred_eos_pos = (predicted == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(pred_eos_pos) > 0:
                    pred_eos_pos = pred_eos_pos[0].item()
                else:
                    pred_eos_pos = seq_len
                eos_pos = min(gt_eos_pos, pred_eos_pos) + 1
                mask[b, eos_pos:] = 0
            batch_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            batch_losses.append(batch_loss.item())
            print(f"Validation Batch {i}, Val Batch Loss: {batch_loss.item():.4f}")

            b, _, seq_size = outputs.shape
            _, predicted = outputs.max(1)
            
            gt = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
            pred = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted]
            gts.extend([[g] for g in gt])
            preds.extend(pred)
            all_images.extend(images.cpu())
            eval_loss += batch_loss * b
            total += b

    bleue, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']
    
    if len(preds) >= 9:
        sample_indices = random.sample(range(len(preds)), 9)
        sampled_preds = [preds[i] for i in sample_indices]
        sampled_gts = [gts[i] for i in sample_indices]
        sampled_images = [all_images[i] for i in sample_indices]
        print("Eval preds (9 random): ", sampled_preds)
        print("Eval gts (9 random): ", sampled_gts)
        wandb.log({
            "eval_predictions": sampled_preds,
            "eval_ground_truths": sampled_gts,
            "eval_images": [wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"Pred: {pred}\nGT: {gt[0]}") for img, pred, gt in zip(sampled_images, sampled_preds, sampled_gts)]
        })
    else:
        print("Eval preds: ", preds)
        print("Eval gts: ", gts)
        wandb.log({
            "eval_predictions": preds,
            "eval_ground_truths": gts,
            "eval_images": [wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"Pred: {pred}\nGT: {gt[0]}") for img, pred, gt in zip(all_images, preds, gts)]
        })

    avg_eval_loss = eval_loss / total
    print(f"Average Validation Loss for Epoch: {avg_eval_loss:.4f}")
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"
    return avg_eval_loss, result


if __name__ == "__main__":
    base_path = '/mnt/dataset/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'FilteredDataSplit.npy'

    config = {
        "encoder_type": "resnet18", # 'resnet18' or 'resnet34'
        "decoder_type": "gru", # 'gru' or 'lstm'
        "apply_teacher_forcing": True,
        "prefix": "/mnt/dataset/image_captioning_dataset/FoodImages",
        "testdata_path": "~/datanew/MIT_small_train_2/test",
        "batch_size": 16,
        "optimizer_type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.05,
        "schedule_sampling_speed": 30,
        "num_epochs": 30,
        "accum_steps": 4,
        "warmup_ep": 3,
        "patience_es": 5,
        "save_dir": "./checkpoints",
    }

    partitions = np.load(splits_path, allow_pickle=True).item()
    
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)

    train(config["num_epochs"], config["prefix"], partitions, metric, config=config)