from model.bert import bert_ATE
from data.dataset import dataset_ATM, remove_duplicates
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import time
import numpy as np


DEVICE = torch.device("mps" if torch.has_mps else "cpu")
pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)

def evl_time(t):
    min, sec= divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model
    
def save_model(model, name):
    torch.save(model.state_dict(), name)
    
laptops_train = remove_duplicates(pd.read_csv("data/laptops_train.csv"))
restaurants_train = remove_duplicates(pd.read_csv("data/restaurants_train.csv"))
twitter_train = remove_duplicates(pd.read_csv("data/twitter_train.csv"))


laptops_train_ds = dataset_ATM(laptops_train, tokenizer)
restaurants_train_ds = dataset_ATM(restaurants_train, tokenizer)
twitter_train_ds = dataset_ATM(twitter_train, tokenizer)


train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])

def create_mini_batch(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    tags_tensors = [s[2] for s in samples]
    tags_tensors = pad_sequence(tags_tensors, batch_first=True)

    pols_tensors = [s[3] for s in samples]
    pols_tensors = pad_sequence(pols_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)
    
    return ids_tensors, tags_tensors, pols_tensors, masks_tensors

train_loader = DataLoader(train_ds, batch_size=10, collate_fn=create_mini_batch, shuffle = True)

def train_model_ATE(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        losses = []
        correct_predictions = 0
        t0 = time.time()
        for data in loader:
            ids_tensors, tags_tensors, _, masks_tensors = data
            ids_tensors = ids_tensors.to(DEVICE)
            tags_tensors = tags_tensors.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            loss = model_ATE(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ATE.step()
            optimizer_ATE.zero_grad()

        current_time = (round(time.time()-t0,3))
        hr, min, sec = evl_time(current_time)
        print('epoch:', epoch, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)         

        save_model(model_ATE, 'bert_ATE.pkl')

train_model_ATE(train_loader, 5)
        
