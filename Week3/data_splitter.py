import pandas as pd
import numpy as np
import sys
import json
from unidecode import unidecode

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file '{file_path}' successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def check(string):
    CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '%', '&', "'", '(', ')', ',', '-', '.', '/', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    NUM_CHAR = len(CHARS)
    IDX2CHAR = {k: v for k, v in enumerate(CHARS)}
    CHAR2IDX = {v: k for k, v in enumerate(CHARS)}
    for char in string:
        if not char in CHAR2IDX:
            print("Error, wrong char: ", char," ord: ", ord(char))
    return string
            

if __name__  == "__main__":
    csv_data = read_csv_file("ImageNameMapping.csv")
    csv_data = csv_data.dropna()
    csv_data = csv_data[~csv_data.apply(lambda row: row.astype(str).str.contains('#NAME?').any(), axis=1)]
    print(csv_data.head())
    print("-------")
    print(csv_data.columns.tolist())
    print("-------")
    l = csv_data.iloc[:, [1, 4]].values
    l[:, 1] = l[:, 1]+".jpg"
    print(l[:, 0])
    t = [x for x in l[:, 0] if not isinstance(x, str)]
    print("t:",t)
    l[:, 0] = [check(unidecode(x).replace("\n", "")) for x in l[:, 0]]
    print("Names: ", l[:, 1])
    print("Shape: ",l.shape[0])
    np.random.shuffle(l)
    
    total_samples = l.shape[0]
    train_size = int(0.8 * total_samples)
    eval_size = int(0.1 * total_samples)
    split_struct = {
        "train":l[:train_size,:],
        "eval":l[train_size:train_size+eval_size,:],
        "test":l[train_size+eval_size:,:],
    }
    print("Train:",split_struct["train"].shape)
    print("Test:",split_struct["test"].shape)
    print("Eval:",split_struct["eval"].shape)

    print("Train:",split_struct["train"][0])
    print("Test:",split_struct["test"][0])
    print("Eval:",split_struct["eval"][0])

    np.save("DataSplit.npy", split_struct)
    json_split_struct = split_struct
    json_split_struct["train"] = json_split_struct["train"].tolist()
    json_split_struct["test"] = json_split_struct["test"].tolist()
    json_split_struct["eval"] = json_split_struct["eval"].tolist()
    with open("DataSplit.json", "w") as json_file:
        json.dump(split_struct, json_file, indent=4)

