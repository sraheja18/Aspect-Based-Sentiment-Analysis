from model.bert import bert_ABSA
from data.dataset import dataset_ABSA, remove_duplicates
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
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)


def evl_time(t):
    min, sec= divmod(t, 60)
    hr, min = divmod(min, 60)
    return int(hr), int(min), int(sec)

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model
    
def save_model(model, name):
    torch.save(model.state_dict(), name)
    

laptops_train_ds = dataset_ABSA(pd.read_csv("data/laptops_train.csv"), tokenizer)
restaurants_train_ds = dataset_ABSA(pd.read_csv("data/restaurants_train.csv"), tokenizer)
twitter_train_ds = dataset_ABSA(pd.read_csv("data/twitter_train.csv"), tokenizer)

def create_mini_batch2(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors = pad_sequence(ids_tensors, batch_first=True)

    segments_tensors = [s[2] for s in samples]
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    label_ids = torch.stack([s[3] for s in samples])
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

    return ids_tensors, segments_tensors, masks_tensors, label_ids


train_ds = ConcatDataset([laptops_train_ds, restaurants_train_ds, twitter_train_ds])
train_loader = DataLoader(train_ds, batch_size=10, collate_fn=create_mini_batch2, shuffle = True)

def train_model_ABSA(loader, epochs):
    all_data = len(loader)
    for epoch in range(epochs):
        losses = []
        current_times = []
        correct_predictions = 0
        t0 = time.time()
        for data in loader:
            
            ids_tensors, segments_tensors, masks_tensors, label_ids = data
            ids_tensors = ids_tensors.to(DEVICE)
            segments_tensors = segments_tensors.to(DEVICE)
            label_ids = label_ids.to(DEVICE)
            masks_tensors = masks_tensors.to(DEVICE)

            loss = model_ABSA(ids_tensors=ids_tensors, label_tensors=label_ids, masks_tensors=masks_tensors, segments_tensors=segments_tensors)
            losses.append(loss.item())
            loss.backward()
            optimizer_ABSA.step()
            optimizer_ABSA.zero_grad()

        current_time = (round(time.time()-t0,3))
        hr, min, sec = evl_time(current_time)
        print('epoch:', epoch, " loss:", np.mean(losses), " hr:", hr, " min:", min," sec:", sec)          
        save_model(model_ABSA, 'bert_ABSA.pkl')

train_model_ABSA(train_loader, 8)