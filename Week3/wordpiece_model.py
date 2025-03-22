import numpy as np
import random
from transformers import ResNetModel, BertTokenizer
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

# Load the BERT tokenizer for wordpiece tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
VOCAB_SIZE = tokenizer.vocab_size  # 30,522 for bert-base-uncased
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
    
        # Caption processing with wordpiece tokenization
        # Tokenize the title using the BERT tokenizer
        # Add [CLS] (start) and [SEP] (end) tokens automatically
        encoded = self.tokenizer(
            title,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)  # Shape: (max_len,)
        
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
            self.decoder = nn.GRU(512, 512, num_layers=1, dropout=0.3)
            self.zero_cell = torch.zeros(1, 512, device=DEVICE)
        elif decoder_type == 'lstm':
            print("lstm decoder")
            self.decoder = nn.LSTM(512, 512, num_layers=1, dropout=0.3)
            self.zero_cell = torch.zeros(1, 512, device=DEVICE)
        else:
            raise ValueError("Unsupported decoder. Choose 'gru' or 'lstm'.")
           
        self.apply_teacher_forcing = apply_teacher_forcing
        self.proj = nn.Linear(512, VOCAB_SIZE)  # Output logits over wordpiece vocab
        self.embed = nn.Embedding(VOCAB_SIZE, 512)  # Embedding for wordpiece tokens
        self.start = torch.tensor(tokenizer.cls_token_id, device=DEVICE)  # [CLS] token as start

    def forward(self, img, titles=None):
        batch_size = img.shape[0]
        feat = self.resnet(img).pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # 1, batch, 512

        if titles is not None and self.training and self.apply_teacher_forcing:  # Teacher forcing
            embeds = self.embed(titles[:, :-1])  # batch, max_len-1, 512
            embeds = embeds.permute(1, 0, 2)     # max_len-1, batch, 512
            if isinstance(self.decoder, nn.LSTM):
                out, _ = self.decoder(embeds, (feat, self.zero_cell.repeat(1, batch_size, 1)))
            else:  # GRU
                out, _ = self.decoder(embeds, feat)
            res = self.proj(out.permute(1, 0, 2))  # batch, max_len-1, VOCAB_SIZE
            return res.permute(0, 2, 1)            # batch, VOCAB_SIZE, max_len-1
        else:  # Sequential generation
            start_embed = self.embed(self.start).repeat(batch_size, 1).unsqueeze(0)  # 1, batch, 512
            inp = start_embed
            hidden = feat
            outputs = []
            for t in range(TEXT_MAX_LEN - 1):
                if isinstance(self.decoder, nn.LSTM):
                    out, (hidden, _) = self.decoder(inp, (hidden, self.zero_cell.repeat(1, batch_size, 1)))
                else:
                    out, hidden = self.decoder(inp, hidden)
                out = self.proj(out.permute(1, 0, 2)).permute(0, 2, 1)  # batch, VOCAB_SIZE, 1
                outputs.append(out)
                _, predicted = out.max(1)  # batch, 1
                inp = self.embed(predicted).permute(1, 0, 2)  # 1, batch, 512
            res = torch.cat(outputs, dim=2)  # batch, VOCAB_SIZE, max_len-1
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
    # Create a unique run ID using timestamp
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_id}"
    
    # Initialize W&B with unique run name
    wandb.init(project="image_captioning", name=run_name, config=config)
    
    encoder_type = config.get("encoder_type", "resnet18")
    decoder_type = config.get("decoder_type", "gru")
    data_train = Data(prefix, partitions['train'], data_aug=False)
    data_valid = Data(prefix, partitions['eval'])
    data_test = Data(prefix, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_valid, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    model = Model(encoder_type=encoder_type, decoder_type=decoder_type, apply_teacher_forcing=config["apply_teacher_forcing"]).to(DEVICE)
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"], config)
    crit = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to compute per-token loss

    # For model saving
    best_val_loss = float('inf')
    base_save_dir = config.get("save_dir", "./checkpoints")
    save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    patience = 5
    epochs_no_improve = 0

    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        train_loss = train_one_epoch(model, optimizer, crit, dataloader_train, accum_steps=config.get("accum_steps", 4), apply_teacher_forcing=config["apply_teacher_forcing"])
        print(f'train loss: {train_loss:.2f}, epoch: {epoch}')
        val_loss, val_metrics = eval_epoch(model, crit, metric, dataloader_valid)
        print(f'valid loss: {val_loss:.2f}, metric: {val_metrics}')

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu1": float(val_metrics.split("BLEU-1:")[1].split("%")[0]) / 100,
            "bleu2": float(val_metrics.split("BLEU2:")[1].split("%")[0]) / 100,
            "rouge_l": float(val_metrics.split("ROUGE-L:")[1].split("%")[0]) / 100,
            "meteor": float(val_metrics.split("METEOR:")[1].split("%")[0]) / 100,
        })

        # Save model if validation loss improves
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
    optimizer.zero_grad()
    for i, (images, titles) in enumerate(tqdm.tqdm(dataloader, desc="Mini batches")):
        images, titles = images.to(DEVICE), titles.to(DEVICE)
        
        # Forward pass
        outputs = model(images, titles)  # outputs: (batch, VOCAB_SIZE, seq_len)
        
        # Compute loss based on the mode
        if apply_teacher_forcing:
            # Teacher forcing mode: compute loss over the entire sequence
            loss = crit(outputs, titles[:, 1:])  # loss: (batch, seq_len)
            loss = loss.mean() / accum_steps  # Average over batch and sequence length
        else:
            # Sequential generation mode: compute loss up to [SEP] token
            batch_size, _, seq_len = outputs.shape
            loss = crit(outputs, titles[:, 1:])  # loss: (batch, seq_len)
            
            # Create a mask to compute loss only up to [SEP]
            mask = torch.ones_like(loss, device=DEVICE)  # (batch, seq_len)
            for b in range(batch_size):
                # Find [SEP] in ground truth
                gt_eos_pos = (titles[b, 1:] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(gt_eos_pos) > 0:
                    gt_eos_pos = gt_eos_pos[0].item()
                else:
                    gt_eos_pos = seq_len
                
                # Find [SEP] in predictions
                _, predicted = outputs[b].max(0)  # predicted: (seq_len,)
                pred_eos_pos = (predicted == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                if len(pred_eos_pos) > 0:
                    pred_eos_pos = pred_eos_pos[0].item()
                else:
                    pred_eos_pos = seq_len
                
                # Use the earlier of the two positions
                eos_pos = min(gt_eos_pos, pred_eos_pos) + 1  # +1 to include [SEP]
                mask[b, eos_pos:] = 0  # Zero out loss after [SEP]
            
            # Apply mask and compute average loss
            loss = (loss * mask).sum() / (mask.sum() + 1e-8) / accum_steps  # Average over non-masked tokens
        
        loss.backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * images.size(0) * accum_steps
        total += images.size(0)
    
    avg_train_loss = train_loss / total
    return avg_train_loss


def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    eval_loss = 0.0
    total = 0
    gts = []
    preds = []
    all_images = []
    with torch.no_grad():
        for images, titles in dataloader:
            images, titles = images.to(DEVICE), titles.to(DEVICE)

            # Forward pass
            outputs = model(images)  # batch, VOCAB_SIZE, seq
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: NaN or Inf in outputs")
            
            # Compute loss (use sequential generation mode's loss for evaluation)
            batch_size, _, seq_len = outputs.shape
            loss = crit(outputs, titles[:, 1:])  # loss: (batch, seq_len)
            mask = torch.ones_like(loss, device=DEVICE)  # (batch, seq_len)
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
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            
            # Track loss and metrics
            b, _, seq_size = outputs.shape
            _, predicted = outputs.max(1)  # batch, seq_len
            
            # Decode predictions and ground truth using the tokenizer
            gt = [tokenizer.decode(title, skip_special_tokens=True) for title in titles]
            pred = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predicted]
            gts.extend([[g] for g in gt])  # Wrap each ground truth in a list for BLEU
            preds.extend(pred)
            
            all_images.extend(images.cpu())
            eval_loss += loss.item() * b
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
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"
    return avg_eval_loss, result


if __name__ == "__main__":
    base_path = '/mnt/dataset/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}DataSplit.npy'

    config = {
            "encoder_type": "resnet18",  # 'resnet18' or 'resnet34'
            "decoder_type": "gru",  # 'gru' or 'lstm'
            "apply_teacher_forcing": True,
            "prefix": "/mnt/dataset/image_captioning_dataset/FoodImages",
            "testdata_path": "~/datanew/MIT_small_train_2/test",
            "batch_size": 32,
            "optimizer_type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "num_epochs": 30,
            "accum_steps": 4,
            "save_dir": "./checkpoints",  
        }

    partitions = np.load(splits_path, allow_pickle=True).item()
    
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)

    train(config["num_epochs"], config["prefix"], partitions, metric, config=config)