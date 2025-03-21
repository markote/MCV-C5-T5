import numpy as np
import random
from transformers import ResNetModel
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

CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '%', '&', "'", '(', ')', ',', '-', '.', '/', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(CHARS)
IDX2CHAR = {k: v for k, v in enumerate(CHARS)}
CHAR2IDX = {v: k for k, v in enumerate(CHARS)}
TEXT_MAX_LEN = 201
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, prefix,  partition, data_aug=False):
        self.prefix = prefix
        self.partition = partition
        self.max_len = TEXT_MAX_LEN
        if data_aug:
            self.img_proc = torch.nn.Sequential(
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224), antialias=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)
        else:
            self.img_proc = torch.nn.Sequential(
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224), antialias=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        title, path = self.partition[idx]
        ## image processing
        img = Image.open(os.path.join(self.prefix, path)).convert('RGB')
        img = self.img_proc(img)
    
        ## caption processing
        # print("Image captioning processing: ")
        # print(title)
        cap_list = list(title)
        final_list = [CHARS[0]]
        final_list.extend(cap_list)
        final_list.extend([CHARS[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([CHARS[2]]*gap)
        cap_idx = [CHAR2IDX[i] for i in final_list]
        # print("final list to idx", final_list)
        # print("final idx", cap_idx)
        # print("final idx in pytorch tensor: ",  torch.tensor(cap_idx, dtype=torch.long))
        # sys.exit(1)
        return img, torch.tensor(cap_idx, dtype=torch.long)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(DEVICE)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) # 1, batch, 512
        start = torch.tensor(CHAR2IDX['<SOS>']).to(DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        for t in range(TEXT_MAX_LEN-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512
    
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 81 (NUM_CHAR == 81)
        res = res.permute(0, 2, 1) # batch, 81, seq
        return res

def optimizer_chooser(model, type_opt):
    if type_opt == "AdamW":
        return optim.AdamW(model.parameters())
    elif type_opt == "Adam":
        return optim.Adam(model.parameters())
    elif type_opt == "SGD":
        return optim.SGD(model.parameters())
    else:
        print("Wrong model")
        sys.exit(1)

def train(epochs, prefix, partitions, metric, config=None):
    data_train = Data(prefix, partitions['train'])
    data_valid = Data(prefix, partitions['eval'])
    data_test = Data(prefix, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    model = Model().to(DEVICE)
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"])
    crit = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
        print(f'train loss: {loss:.2f}, metric: {res}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid)
        print(f'valid loss: {loss:.2f}, metric: {res}')
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test)
    print(f'test loss: {loss:.2f}, metric: {res}')
    
def train_one_epoch(model, optimizer, crit, metric, dataloader):
    model.train()    
    # Training loop
    train_loss = 0.0
    total = 0
    gts = []
    preds = []
    for images, titles in dataloader:
        # print("images shape: ", images.shape)
        # print("titles shape: ", titles.shape)
        # print("titles: ", titles)
        # sys.exit(1)
        images, titles = images.to(DEVICE), titles.to(DEVICE) # titles should be a tensor of shape (batch, num_seq vector) with each element being between [0, NUM_CHAR-1]
        
        # Forward pass
        outputs = model(images) # batch, NUM_CHAR, seq
        loss = crit(outputs, titles).sum() 
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss and metrics
        b,_,seq_size = outputs.shape

        _, predicted = outputs.max(1) # we are interested on the pos of the max logits for the NUM_CHAR dimension so basically the predicted char (batch, num_seq vector) with each element being between [0, NUM_CHAR-1] 
        idx_titles_T = titles.T  # Shape: (seq_len, batch_size)
        idx_predicted_T = predicted.T # Shape: (seq_len, batch_size)

        # print("title shape: ", titles.shape)
        # print("pred shape: ", predicted.shape)
        gt = [["".join([IDX2CHAR[idx.item()] for idx in seq])] for seq in titles] # gt muyst be [gt_ref1, gtref2,] for each prediction
        pred = ["".join([IDX2CHAR[idx.item()] for idx in seq]) for seq in predicted]
        gts = gts + gt
        preds = preds + pred
        train_loss += loss.item() * b # compute the avg loss by sum of all seq chars
        total += b
        
    bleue, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']
    
    # Calculate training metrics
    avg_train_loss = train_loss / total
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"

    return avg_train_loss, result

def eval_epoch(model, crit, metric, dataloader):
    model.eval()
    total = 0
    eval_loss = 0.0
    total = 0
    gts = []
    preds = []
    with torch.no_grad():
        for images, titles in dataloader:
            images, titles = images.to(DEVICE), titles.to(DEVICE) # titles should be a tensor of shape (batch, num_seq vector) with each element being between [0, NUM_CHAR-1]

            # Forward pass
            outputs = model(images) # batch, NUM_CHAR, seq
            loss = crit(outputs, titles).sum() 

            # Track loss and metrics
            b,_,seq_size = outputs.shape

            _, predicted = outputs.max(1) # we are interested on the pos of the max logits for the NUM_CHAR dimension so basically the predicted char (batch, num_seq vector) with each element being between [0, NUM_CHAR-1] 

            gt = [["".join([IDX2CHAR[idx.item()] for idx in seq])] for seq in titles]
            pred = ["".join([IDX2CHAR[idx.item()] for idx in seq]) for seq in predicted]
            
            gts = gts + gt
            preds = preds + pred
            eval_loss += loss.item() * b # compute the avg loss by sum of all seq chars
            total += b

    bleue, rouge, meteor = metric
    bleu1 = bleu.compute(predictions=preds, references=gts, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=preds, references=gts, max_order=2)["bleu"]
    res_r = rouge.compute(predictions=preds, references=gts)['rougeL']
    res_m = meteor.compute(predictions=preds, references=gts)['meteor']
    
    # Calculate training metrics
    avg_eval_loss = eval_loss / total
    print("Eval preds: ", preds)
    print("Eval gts: ", gts)
    result = f"BLEU-1:{bleu1*100:.1f}%, BLEU2:{bleu2*100:.1f}%, ROUGE-L:{res_r*100:.1f}%, METEOR:{res_m*100:.1f}%"

    return avg_eval_loss, result

if __name__ == "__main__":

    base_path = '/ghome/c5mcv05/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}DataSplit.npy'

    config = {
            "prefix": "/ghome/c5mcv05/image_captioning_dataset/FoodImages",
            "testdata_path": "~/datanew/MIT_small_train_2/test",
            "batch_size": 32,
            "optimizer_type": "SGD",
            "num_epochs": 1,
        }

    partitions = np.load(splits_path, allow_pickle=True).item()
    
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    metric = (bleu, rouge, meteor)

    train(config["num_epochs"], config["prefix"], partitions, metric, config=config)


