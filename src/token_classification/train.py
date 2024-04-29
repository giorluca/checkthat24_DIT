from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import TrainingDataset
from sequence_aligner.containers import TrainingBatch
import json
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
from datasets import Dataset as hf_Dataset   
from torch.utils.data import Dataset
import pandas as pd
import os
import sys
sys.path.append('/home/pgajo/food/src')
#from utils_food import save_local_model
import re
from datetime import datetime
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# train_data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold.json'
train_data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold_binary.json'

with open(train_data_path, 'r', encoding='utf8') as f:
    dataset_raw = json.load(f)

df = pd.DataFrame(dataset_raw)
dataset = hf_Dataset.from_pandas(df[['lang', 'annotations', 'text', 'article_id']])#.filter(lambda example: example["text"] is not None) # some samples have no text and cannot be tokenized so we filter them out
# dataset = dataset.filter(lambda example: example['lang'] == 'en')
split_ratio = 0.2
split_seed = 42
datadict = dataset.train_test_split(split_ratio, seed=split_seed)
train_data = datadict['train']
val_data = datadict['test']


# labels = ["Appeal_to_Authority", "Appeal_to_Popularity", "Appeal_to_Values", "Appeal_to_Fear-Prejudice", "Flag_Waving",
#                              "Causal_Oversimplification", "False_Dilemma-No_Choice", "Consequential_Oversimplification", "Straw_Man",
#                              "Red_Herring", "Whataboutism", "Slogans", "Appeal_to_Time", "Conversation_Killer", "Loaded_Language",
#                              "Repetition", "Exaggeration-Minimisation", "Obfuscation-Vagueness-Confusion", "Name_Calling-Labeling", "Doubt",
#                              "Guilt_by_Association", "Appeal_to_Hypocrisy", "Questioning_the_Reputation"]
# label_set = LabelSet(labels=labels)
labels = ["persuasion"]
label_set = LabelSet(labels=labels)

model_name = 'bert-base-multilingual-cased'
# model_name = 'xlm-roberta-base'
# model_name = 'microsoft/mdeberta-v3-base'
model_name_simple = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TrainingDataset(data=train_data,tokenizer=tokenizer,label_set=label_set, tokens_per_batch=128)#.filter(lambda example: example["labels"] is not None)
val_dataset = TrainingDataset(data=val_data,tokenizer=tokenizer,label_set=label_set, tokens_per_batch=128)

train_loader = DataLoader(train_dataset, collate_fn=TrainingBatch, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, collate_fn=TrainingBatch, batch_size=64, shuffle=False)

model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=len(train_dataset.label_set.ids_to_label.values())
                                                        )
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
            attention_mask = batch['attention_masks'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            eval_loss += loss.item()

            preds.extend(np.argmax(outputs['logits'].detach().cpu().numpy(), axis=2).flatten())
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
        'attention_mask': batch['attention_masks'].to(device),
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

models_dir = '/home/pgajo/checkthat24/checkthat24_DIT/models'
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



# checklist = ['.', '...', '\n\n']
# for sample in dataset:#datadict['train']:
#     for annotation in sample['annotations']:
#         start = annotation['start']
#         end = annotation['end']
#         text = sample['text']
#         # text = re.sub(r'\n+', ' ', text)
#         slice = text[start:end+1]
#         print('--------------')
#         print('left:', text[:start][-20:])
#         print('>>> slice:', slice)
#         print('right:', text[end+1:][:20])
#         print('--------------')

        # for char in checklist:
        #     if char in slice and slice[-1] != '.':
        #         print(text[start:end])
        #         print('------------')
