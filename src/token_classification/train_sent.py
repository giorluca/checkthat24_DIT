import json
from sequence_aligner.labelset import LabelSet
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
from icecream import ic
import re
from datetime import datetime
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# train_data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold.json'
train_data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold/train_gold_sentences.json'

with open(train_data_path, 'r', encoding='utf8') as f:
    dataset_raw = json.load(f)

df_raw = pd.DataFrame(dataset_raw)  

df_pos = df_raw[df_raw['annotations'].apply(lambda x: len(x) > 0)]
ic(df_pos)
df_neg = df_raw[df_raw['annotations'].apply(lambda x: x == [])].sample(len(df_pos))
ic(df_neg)
df = pd.concat([
    df_pos,
    # df_neg
    ]
    )
ic(len(df_pos))
ic(len(df_neg))
ic(len(df))
dataset = Dataset.from_pandas(df[['lang', 'annotations', 'text', 'article_id']]).filter(lambda example: example["text"] is not None) # some samples have no text and cannot be tokenized so we filter them out
# dataset = dataset.filter(lambda example: example['lang'] == 'en')
split_ratio = 0.2
split_seed = 42
datadict = dataset.train_test_split(split_ratio, seed=split_seed)

mappings = [
    {'pattern': r'(?<!\s)([^\w\s])|([^\w\s])(?!\s)', 'target': ' placeholder '},
    {'pattern': r'\s+', 'target': ' '},
    ]

def sub_shift_spans(text, ent_list, mappings = [], diacritics = 'àèìòùáéíóúÀÈÌÒÙÁÉÍÓÚ'):
    for mapping in mappings:
        adjustment = 0
        pattern = re.compile(mapping['pattern'])
        for match in re.finditer(pattern, text):
            match_index = match.start() + adjustment
            match_contents = match.group()
            if match_contents not in diacritics:
                subbed_text = mapping['target'].replace('placeholder', match_contents)
            else:
                subbed_text = match_contents
            len_diff = len(subbed_text) - len(match_contents)
            text = text[:match_index] + subbed_text + text[match_index + len(match_contents):]
            
            for ent in ent_list:
                if ent['start'] <= match_index and ent['end'] > match_index:
                    ent['end'] += len_diff
                if ent['start'] > match_index:
                    ent['start'] += len_diff
                    ent['end'] += len_diff
            adjustment += len_diff
    # for ent in ent_list:
    #     ic(text[ent['start']:ent['end']])
    return text, ent_list

model_name = 'bert-base-multilingual-cased'
# model_name = 'xlm-roberta-base'
# model_name = 'microsoft/mdeberta-v3-base'
model_name_simple = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(samples, tokenizer):
    texts = []
    ents_list = []
    for text, annotation_list in zip(samples['text'], samples['annotations']):
        for annotation in annotation_list:
            out_text, out_ents = sub_shift_spans(text, annotation, mappings=mappings)

            out_text_split = re.findall(r'[^\s]+', out_text)
            labels_words = []
            for match in out_text_split:
                if match.start < out_ents['start']:
                    labels_words.append(labels.labels_to_id['O'])
                elif match.start == out_ents['start']:
                    labels_words.append(labels.labels_to_id['B-' + out_ents['tag']])
                elif i > out_ents['start'] and i < out_ents['end']:
                    labels_words.append(labels.labels_to_id['I-' + out_ents['tag']])
                else:
                    labels_words.append(labels.labels_to_id['O'])
            
            ic(labels_words)

        # output = tokenizer(out_text, return_tensors = 'pt', padding = 'longest', truncation = True)
        # index_tok_start = output.token_to_char(annotation['start'])
        # index_tok_end = output.token_to_char(annotation['end'])
        # labels = []
        # tag = annotation['tag'] # IT'S NOT JUST ONE TAG IN ONE SENTENCE IT CAN BE MORE THAN ONE
        # for i in range(len(output['input_ids'].squeeze())):
        #     if i < index_tok_start:
        #         labels.append(labels.labels_to_id['O'])
        #     elif i == index_tok_start:
        #         labels.append(labels.labels_to_id['B-' + tag])
        #     elif i > index_tok_start and i < index_tok_end:
        #         labels.append(labels.labels_to_id['I-' + tag])
        #     else:
        #         labels.append(labels.labels_to_id['O'])
        
        # output['labels'] = torch.tensor(labels)


    return samples

datadict = datadict.map((lambda x: preprocess(x, tokenizer)), batched=True)

train_data = datadict['train']
val_data = datadict['test']

labels = LabelSet(labels=df_pos['annotations'].apply(lambda x: x[0]['tag']).tolist())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=len(train_data),
                                                        label2id=labels.labels_to_id,
                                                        id2label=labels.ids_to_label,
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
