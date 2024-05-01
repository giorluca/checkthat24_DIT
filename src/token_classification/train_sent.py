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
    df_neg
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

labels_model = LabelSet(labels=set(df_pos['annotations'].apply(lambda x: x[0]['tag']).tolist()))

mappings = [
    {'pattern': r'(?<!\s)([^\w\s])|([^\w\s])(?!\s)', 'target': ' placeholder '},
    {'pattern': r'\s+', 'target': ' '},
    ]

import unicodedata
import copy

def sub_shift_spans(text, ents, mappings = []):
    original = copy.deepcopy(text)
    for mapping in mappings:
        adjustment = 0
        pattern = re.compile(mapping['pattern'])
        for match in re.finditer(pattern, text):
            match_index = match.start() + adjustment
            match_contents = match.group()
            if all(unicodedata.category(char).startswith('P') for char in match_contents):
                subbed_text = mapping['target'].replace('placeholder', match_contents)
            else:
                subbed_text = match_contents
            len_diff = len(subbed_text) - len(match_contents)
            text = text[:match_index] + subbed_text + text[match_index + len(match_contents):]

            if isinstance(ents, list):
                for ent in ents:
                    if ent['start'] <= match_index and ent['end'] > match_index:
                        ent['end'] += len_diff
                    if ent['start'] > match_index:
                        ent['start'] += len_diff
                        ent['end'] += len_diff
            elif isinstance(ents, dict):
                if ents['start'] <= match_index and ents['end'] > match_index:
                    ents['end'] += len_diff
                if ents['start'] > match_index:
                    ents['start'] += len_diff
                    ents['end'] += len_diff

            adjustment += len_diff
    # for ent in ent_list:
    #     ic(text[ent['start']:ent['end']])
    return text, ents

# model_name = 'bert-base-multilingual-cased'
# model_name = 'xlm-roberta-base'
model_name = 'microsoft/mdeberta-v3-base'
model_name_simple = model_name.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(model_name)

def span_to_words_annotation(samples, target_tag = ''):
    samples_new = []
    for i, (text, annotation_list) in enumerate(zip(samples['text'], samples['annotations'])):
        labels_text = []
        tokens = []
        for j, annotation in enumerate(annotation_list):
            text_subshifted, ents = sub_shift_spans(text, annotation, mappings=mappings)
            text_subshifted_matches = re.finditer(r'[^\s]+', text_subshifted)
            labels_words = []
            first = 1
            for k, match in enumerate(text_subshifted_matches):
                if j == 0:
                    tokens.append(match.group())
                if match.start() < ents['start']:
                    labels_words.append(labels_model.labels_to_id['O'])
                elif match.start() >= ents['start'] and match.end() <= ents['end']:
                    if first == 1:
                        labels_words.append(labels_model.labels_to_id['B-' + ents['tag']])
                        first = 0
                    elif first == 0:
                        labels_words.append(labels_model.labels_to_id['I-' + ents['tag']])
                else:
                    labels_words.append(labels_model.labels_to_id['O'])
            labels_text.append({'labels': labels_words, 'tag': annotation['tag']})

        # if the training sample has no tags that we need, we just produce a 0s list
        if target_tag not in [labels['tag'] for labels in labels_text]:
            labels = [0] * len(tokens)
        # if the training sample has tags we need, we first exclude the label lists whose tags don't match
        # and then we merge the label lists that have tags that match the target tag
        else:
            labels = [max(values) for values in zip(*[labels['labels'] for labels in labels_text if labels['tag'] == target_tag])]
        samples_new.append({
            'id': i,
            'ner_tags': labels,
            'tokens': tokens,
        })
    return samples_new

from collections import defaultdict

def dict_of_lists(lst_of_dicts):
    result = defaultdict(list)
    for d in lst_of_dicts:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)

def tokenize_and_align_labels(examples):
    examples = dict_of_lists(examples)
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='longest', return_tensors='pt')

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = [tokenized_inputs.token_to_word(i, j) for j in range(len(tokenized_inputs['input_ids'][i]))]  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs

target_tags = [(i, el.strip()) for i, el in enumerate(open('/home/pgajo/checkthat24/checkthat24_DIT/persuasion_techniques.txt').readlines())]

for tt in target_tags:

    datadict = datadict.map((lambda x: tokenize_and_align_labels(span_to_words_annotation(x, target_tag=tt[1]))), batched=True)

    columns = [
                'input_ids',
                'token_type_ids',
                'attention_mask',
                'labels'
                ]
    
    datadict.set_format('torch', columns = columns)

    train_data = datadict['train']
    val_data = datadict['test']

    from transformers import DataCollatorForTokenClassification 

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding = 'longest')

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=data_collator)

    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                            num_labels=len(labels_model.ids_to_label.values()),
                                                            label2id=labels_model.labels_to_id,
                                                            id2label=labels_model.ids_to_label,
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
                attention_mask = batch['attention_mask'].to(device)
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

    models_dir = '/home/pgajo/checkthat24/checkthat24_DIT/models'
    model_save_name = f'{model_name_simple}_{tt[0]}_ME{epochs}_target={tt[1]}_{date_time}'
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
