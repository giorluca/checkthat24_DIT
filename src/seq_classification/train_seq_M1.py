import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from datasets import Dataset as Dataset   
import pandas as pd
import os
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import save_local_model
from datetime import datetime
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
from icecream import ic

train_data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold_sentences_mine.json'

with open(train_data_path, 'r', encoding='utf8') as f:
    dataset_raw = json.load(f)

langs = []
for el in dataset_raw:
    langs.append(el['lang'])
langs = set(langs)

df = pd.DataFrame(dataset_raw)
df_pos = df[df['label'] == 1]
df_neg = pd.DataFrame()
for lang in langs:
    df_lang = df[df['lang'] == lang]
    df_pos_lang = df_pos[df_pos['lang'] == lang]
    df_lang_sampled = df_lang.sample(len(df_pos_lang))
    df_neg = pd.concat([df_neg, df_lang_sampled])

df = pd.concat([df_pos, df_neg], axis=0)
ic(df_pos['lang'].value_counts())
ic(df_neg['lang'].value_counts())
ic(df['lang'].value_counts())
dataset = Dataset.from_pandas(df[['lang', 'annotations', 'text', 'article_id', 'label']])#.filter(lambda example: example["text"] is not None) # some samples have no text and cannot be tokenized so we filter them out

# model_name = 'bert-base-multilingual-cased'
# model_name = 'xlm-roberta-base'
model_name = 'microsoft/mdeberta-v3-base'
model_name_simple = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # labels = []
    # for example in examples['annotations']:
    #     labels.append(int(bool(example[0]['tag'])))
        
    output = tokenizer(examples["text"],
                                truncation=True,
                                padding = 'longest',
                                return_tensors='pt'
                                )
    output['labels'] = torch.tensor(examples['label'])
    return output

split_ratio = 0.2
split_seed = 42
batch_size = 16
datadict = dataset.train_test_split(split_ratio, seed=split_seed)
datadict = datadict.map(preprocess_function,
                        batch_size=batch_size,
                        batched=True
                        )
columns = [
            'input_ids',
            'token_type_ids',
            'attention_mask',
            'labels'
            ]
datadict.set_format('torch', columns = columns)

train_data = datadict['train']#(range(100))
val_data = datadict['test']#(range(100))

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding = 'longest')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

for el in train_loader:
    print(el)
    break

model = AutoModelForSequenceClassification.from_pretrained(model_name)

lr = 5e-5
optimizer = AdamW(model.parameters(), lr=lr)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epochs = 3

model.to(device)

def evaluate():
    model.eval()
    eval_loss = 0
    preds = []
    out_label_ids = []
 
    progbar_val = tqdm(val_loader)
    for i, batch in enumerate(progbar_val):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            eval_loss += loss.item()

            preds.extend(np.argmax(outputs['logits'].detach().cpu().numpy(), axis=1).flatten())
            out_label_ids.extend(labels.detach().cpu().numpy().flatten())

            val_loss_tmp = round(eval_loss / (i + 1), 4)
            progbar_val.set_postfix({'Eval loss':val_loss_tmp})

    preds = np.array(preds)
    out_label_ids = np.array(out_label_ids)

    results = classification_report(out_label_ids, preds, output_dict=True)

    return results

for epoch in range(epochs):
    model.train()
    train_loss = 0
    progbar_train = tqdm(train_loader)
    for i, batch in enumerate(progbar_train):
        optimizer.zero_grad()
        inputs = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device),
        }

        outputs = model(**inputs)

        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()

        train_loss_tmp = round(train_loss / (i + 1), 4)
        progbar_train.set_postfix({'Train loss':train_loss_tmp})

    results = evaluate()
    print(results)

models_dir = '/home/pgajo/checkthat24/checkthat24_DIT/models/M1'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model_save_name = f'{model_name_simple}_ME{epochs}_{date_time}'
model_save_dir = os.path.join(models_dir, model_save_name)
save_local_model(model_save_dir, model, tokenizer)

results['train_data_path'] = train_data_path

with open(os.path.join(model_save_dir, 'results.json'), 'w', encoding='utf8') as f:
    json.dump(results, f, ensure_ascii = False)

info = {
    'model_name': model_name,
    'epochs': epochs,
    'lr': lr,
    'len(train_data)': len(train_data),
    'len(val_data)': len(val_data),
    'split_ratio': split_ratio,
    'split_seed': split_seed,
    'train_data_path': train_data_path,

}

with open(os.path.join(model_save_dir, 'info.json'), 'w', encoding='utf8') as f:
    json.dump(info, f, ensure_ascii = False)