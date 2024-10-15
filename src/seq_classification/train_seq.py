import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
sys.path.append('./src')
from utils_checkthat import save_local_model
from token_classification.train_sent import dict_of_lists
from datetime import datetime
#from icecream import ic

def evaluate(model, val_loader, device):
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

def tokenize_sequence_classification(examples, tokenizer):
    annotations = examples['annotations']
    data = dict_of_lists(examples['data'])
    output = tokenizer(data["text"],
                                truncation=True,
                                padding = 'longest',
                                return_tensors='pt'
                                )
    
    output['labels'] = torch.tensor(data['label'])
    output.update(data)
    output['annotations'] = annotations
    return output

def main():
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    train_data_path = './data/formatted/train_sentences.json'

    with open(train_data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    langs = []
    for sample in data:
        langs.append(sample['data']['lang'])
    langs = set(langs)

    df = pd.DataFrame(data)
    df_sampled = pd.DataFrame()
    for lang in langs:
        df_lang = df[df['data'].apply(lambda x: x['lang'] == lang)]
        df_pos_lang = df_lang[df_lang['data'].apply(lambda x: x['label'] == 1)]
        df_neg_lang = df_lang[df_lang['data'].apply(lambda x: x['label'] == 0)]
        if len(df_neg_lang) > len(df_pos_lang):
            df_neg_lang = df_neg_lang.sample(len(df_pos_lang))
            print('len(df_neg_lang) > len(df_pos_lang)')
        df_lang_sampled = pd.concat([df_pos_lang, df_neg_lang])
        df_sampled = pd.concat([df_sampled, df_lang_sampled])
    
    df_sampled_pos = df_sampled[df_sampled['data'].apply(lambda x: x['label'] == 1)]
    df_sampled_neg = df_sampled[df_sampled['data'].apply(lambda x: x['label'] == 0)]
    
    ic(df_sampled_pos['data'].apply(lambda x: x['lang']).value_counts())
    ic(df_sampled_neg['data'].apply(lambda x: x['lang']).value_counts())
    ic(df_sampled['data'].apply(lambda x: x['lang']).value_counts())
    dataset = Dataset.from_pandas(df_sampled)

    # model_name = 'bert-base-multilingual-cased'
    # model_name = 'xlm-roberta-base'
    model_name = 'microsoft/mdeberta-v3-base'
    model_name_simple = model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    split_ratio = 0.2
    split_seed = 42
    batch_size = 16
    datadict = dataset.train_test_split(split_ratio, seed=split_seed)
    datadict = datadict.map(lambda x: tokenize_sequence_classification(x, tokenizer),
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

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    lr = 5e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 10

    model.to(device)

    best_f1_score = 0
    best_model = model
    results = {'results': []}
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
        results_entry = evaluate(model, val_loader, device)
        results_entry['epoch'] = epoch
        results['results'].append(results_entry)
        print(results['results'][-1])

        current_f1_score = results['results'][-1]['macro avg']['f1-score']
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_model = model
            best_epoch = epoch
            print(f'Best model updated: current epoch macro f1 = {current_f1_score}')

    results['best_epoch'] = best_epoch + 1

    models_dir = './models/M1'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_save_name = f'{model_name_simple}_ME{epochs}_{date_time}'
    model_save_dir = os.path.join(models_dir, model_save_name)
    save_local_model(model_save_dir, best_model, tokenizer)

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

if __name__ == "__main__":
    main()