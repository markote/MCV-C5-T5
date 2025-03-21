import numpy as np
import random
from transformers import ResNetModel
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import pandas as pd
import evaluate
import sys
import os

CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(CHARS)
IDX2CHAR = {k: v for k, v in enumerate(chars)}
CHAR2IDX = {v: k for k, v in enumerate(chars)}
TEXT_MAX_LEN = 201
DEVICE = 'cuda'

class Data(Dataset):
    def __init__(self, prefix,  partition, data_aug=False):
        self.prefix = prefix
        self.partition = partition
        self.num_captions = 5
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
        caption = item.caption.reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_list = list(caption)
        final_list = [CHARS[0]]
        final_list.extend(cap_list)
        final_list.extend([CHARS[1]])
        gap = self.max_len - len(final_list)
        final_list.extend([CHARS[2]]*gap)
        cap_idx = [char2idx[i] for i in final_list]
        return img, cap_idx

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
        start = torch.tensor(char2idx['<SOS>']).to(DEVICE)
        start_embed = self.embed(start) # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # 1, batch, 512
        inp = start_embeds
        hidden = feat
        for t in range(TEXT_MAX_LEN-1): # rm <SOS>
            out, hidden = self.gru(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0) # N, batch, 512
    
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res) # batch, seq, 80 (NUM_CHAR == 80)
        res = res.permute(0, 2, 1) # batch, 80, seq
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

def train(epochs, data, partitions, config=None):
    data_train = Data(data, partitions['train'])
    data_valid = Data(data, partitions['eval'])
    data_test = Data(data, partitions['test'])
    dataloader_train = DataLoader(data_train, batch_size=config["batch_size"], pin_memory=True, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    dataloader_test = DataLoader(data_test, batch_size=config["batch_size"], pin_memory=True, shuffle=False, num_workers=8)
    model = Model().to(DEVICE)
    model.train()
    optimizer = optimizer_chooser(model, config["optimizer_type"])
    crit = nn.CrossEntropyLoss()
    metric = Metric()
    for epoch in range(epochs):
        loss, res = train_one_epoch(model, optimizer, crit, metric, dataloader_train)
        print(f'train loss: {loss:.2f}, metric: {res:.2f}, epoch: {epoch}')
        loss_v, res_v = eval_epoch(model, crit, metric, dataloader_valid)
        print(f'valid loss: {loss:.2f}, metric: {res:.2f}')
    loss_t, res_t = eval_epoch(model, crit, metric, dataloader_test)
    print(f'test loss: {loss:.2f}, metric: {res:.2f}')
    
def train_one_epoch(model, optimizer, crit, metric, dataloader):
    model.train()    
    # Training loop
    train_loss = 0.0
    correct = 0.0
    total = 0
    for images, titles in dataloader:
        images, titles = images.to(device), titles.to(device) # titles should be a tensor of shape (batch, num_seq vector) with each element being between [0, NUM_CHAR-1]
        
        # Forward pass
        outputs = model(images) # batch, NUM_CHAR, seq

        loss = 0.0
        loss = crit(outputs, titles).sum() 
        # print(loss)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        b,_,seq_size = outputs.shape
        train_loss += loss.item() * b * seq_size # total loss avg for each char
        _, predicted = outputs.max(1) # we are interested on the pos of the max logits for the NUM_CHAR dimension so basically the predicted char
        correct += (predicted == titles).sum().item() # avg accuracy for each char
        total += titles.shape[0]*titles.shape[1]

    # Calculate training metrics
    avg_train_loss = train_loss / total
    train_accuracy = correct / total
    
    return avg_train_loss, train_accuracy

def eval_epoch(model, crit, metric, dataloader):
    model.eval()    
    # Training loop
    eval_loss = 0.0
    correct = 0.0
    total = 0

    with torch.no_grad():
        for images, titles in dataloader:
            images, titles = images.to(device), titles.to(device) # titles should be a tensor of shape (batch, num_seq vector) with each element being between [0, NUM_CHAR-1]

            # Forward pass
            outputs = model(images) # batch, NUM_CHAR, seq

            loss = 0.0
            loss = crit(outputs, titles).sum() 
            # print(loss)

            # Track loss and accuracy
            b,_,seq_size = outputs.shape
            eval_loss += loss.item() * b * seq_size # total loss avg for each char
            _, predicted = outputs.max(1) # we are interested on the pos of the max logits for the NUM_CHAR dimension so basically the predicted char
            correct += (predicted == titles).sum().item() # avg accuracy for each char
            total += titles.shape[0]*titles.shape[1]

    # Calculate eval metrics
    avg_eval_loss = eval_loss / total
    train_accuracy = correct / total
    
    return avg_eval_loss, train_accuracy


if __name__ == "__main__":

    base_path = '/ghome/c5mcv05/image_captioning_dataset/'
    img_path = f'{base_path}FoodImages/'
    splits_path = f'{base_path}DataSplit.npy'

    partitions = np.load(splits_path, allow_pickle=True).item()


