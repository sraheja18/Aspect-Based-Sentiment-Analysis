from model.bert import bert_ATE, bert_ABSA
from transformers import BertTokenizer
import torch
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

DEVICE = torch.device("mps" if torch.has_mps else "cpu")
pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
lr = 2e-5
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
optimizer_ATE = torch.optim.Adam(model_ATE.parameters(), lr=lr)
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
optimizer_ABSA = torch.optim.Adam(model_ABSA.parameters(), lr=lr)

polarities = {0:"Negative", 1:"Neutral", 2:"Positive"}

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model

model_ATE = load_model(model_ATE, 'bert_ATE.pkl')
model_ABSA = load_model(model_ABSA, 'bert_ABSA.pkl')

def predict_model_ATE(sentence, tokenizer):
    word_pieces = []
    tokens = tokenizer.tokenize(sentence)
    word_pieces += tokens

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None)
        _, predictions = torch.max(outputs, dim=2)
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs

def predict_model_ABSA(sentence, aspect, tokenizer):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor)
        outputs = torch.softmax(outputs, dim=1)
        _,predictions = torch.max(outputs,dim=1)
    
    return word_pieces, predictions, outputs

def ATE_ABSA(text):
    terms = []
    word = ""

    d = {}
    x, y, z = predict_model_ATE(text, tokenizer)
    for i in range(len(y)):
        if y[i] == 1:
            if len(word) != 0:
                terms.append(word.replace(" ##",""))
            word = x[i]
        if y[i] == 2:
            word += (" " + x[i])
            
    
    if len(word) != 0:
            terms.append(word.replace(" ##",""))
            
    final = []
    if len(terms) != 0:
        for i in terms:
            d = {}
            _, c, p = predict_model_ABSA(text, i, tokenizer)
            d["term"] = i
            d['class'] = polarities[int(c)]
            d['probability'] = round(float(p[0][int(c)]),3)
            final.append(d.copy())
    return final

@app.route("/") 
def index():
    return render_template("index.html")

@app.route("/analyze",methods=['POST'])
def analyze_text():
    if request.method == "POST":
        text = request.form["text"]
        analyzed_text = ATE_ABSA(text)
        return render_template("index.html", text=text, analyzed_text=analyzed_text)
    return render_template("index.html")

if __name__=='__main__':
    app.run(debug=True)



