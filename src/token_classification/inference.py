import json
import sys
sys.path.append('./src')
from token_classification.train_sent import span_to_words_annotation, dict_of_lists, tokenize_token_classification, compute_metrics
from seq_classification.train_seq import tokenize_sequence_classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, DataCollatorWithPadding, DataCollatorForTokenClassification
import pandas as pd
from sequence_aligner.labelset import LabelSet
from datasets import Dataset
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

def main():

    json_path = './data/formatted/dev_sentences.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    path_m1 = './models/M1/mdeberta-v3-base_ME3_2024-05-08-20-23-31'
    m1_simple = path_m1.split('/')[-1]
    tokenizer_m1 = AutoTokenizer.from_pretrained(path_m1)

    df_seq = pd.DataFrame(data)
    dataset_seq = Dataset.from_pandas(df_seq[['lang', 'annotations', 'text', 'article_id', 'label']])
    binary_dataset_seq = dataset_seq.map(lambda x: tokenize_sequence_classification(x, tokenizer_m1), batched=True, batch_size=None)

    columns = [ 'input_ids', 'token_type_ids', 'attention_mask', 'labels']
    binary_dataset_seq.set_format('torch', columns = columns)
    data_collator_seq = DataCollatorWithPadding(tokenizer=tokenizer_m1, padding='longest')
    val_loader_seq = DataLoader(binary_dataset_seq, batch_size=16, shuffle=False, collate_fn=data_collator_seq)

    mappings = [
        {'pattern': r'(?<!\s)([^\w\s])|([^\w\s])(?!\s)', 'target': ' placeholder '},
        {'pattern': r'\s+', 'target': ' '},
        ]
    
    target_tags = [(i, el.strip()) for i, el in enumerate(open('./data/persuasion_techniques.txt').readlines())]
    for i, tt in enumerate(target_tags):
        print(f'Infering with m2 no. {i} of {len(target_tags)} for {tt} persuasion technique...')
        labels_model = LabelSet(labels=[tt[1]])
        
        df_list_binary = span_to_words_annotation(dict_of_lists(data), target_tag=tt[1], mappings=mappings, labels_model=labels_model)
        df_binary = pd.DataFrame(df_list_binary)
        binary_dataset = Dataset.from_pandas(df_binary[['id', 'ner_tags', 'tokens']])
        
        model_dir_m2 = './models/M2'

        for model_dir in os.listdir(model_dir_m2):
            model_number = int(model_dir.split('_')[1])
            if model_number == i:
                path_m2 = os.path.join(model_dir_m2, model_dir)
                break

        m2_simple = path_m2.split('/')[-1]
        tokenizer_m2 = AutoTokenizer.from_pretrained(path_m2)

        binary_dataset_token = binary_dataset.map(lambda x: tokenize_token_classification(x, tokenizer_m2), batched=True, batch_size=None)        
        binary_dataset_token.set_format('torch', columns = columns)
        data_collator_token = DataCollatorForTokenClassification(tokenizer=tokenizer_m2, padding='longest')
        val_loader_token = DataLoader(binary_dataset_token, batch_size=16, shuffle=False, collate_fn=data_collator_token)
        
        m1 = AutoModelForSequenceClassification.from_pretrained(path_m1)
        m2 = AutoModelForTokenClassification.from_pretrained(path_m2,
                                                                num_labels=len(labels_model.ids_to_label.values()),
                                                                label2id=labels_model.labels_to_id,
                                                                id2label=labels_model.ids_to_label,
                                                                )
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        m1.to(device)
        m1.eval()
        m2.to(device)
        m2.eval()
        eval_loss = 0
        preds_list = []
        golds_list = []
        label_list = [value for value in labels_model.ids_to_label.values()]

        progbar_val = tqdm(zip(val_loader_seq, val_loader_token))
        for i, (batch_seq, batch_token) in enumerate(progbar_val):
            with torch.no_grad():
                inputs_seq = {
                    'input_ids': batch_seq['input_ids'].to(device),
                    'token_type_ids': batch_seq['token_type_ids'].to(device),
                    'attention_mask': batch_seq['attention_mask'].to(device),
                    'labels': batch_seq['labels'].to(device),
                }

                out_1 = m1(**inputs_seq)

                preds_1 = torch.argmax(out_1.logits, dim=1)

                inputs_token = {
                    'input_ids': batch_token['input_ids'].to(device),
                    'token_type_ids': batch_token['token_type_ids'].to(device),
                    'attention_mask': batch_token['attention_mask'].to(device),
                    'labels': batch_token['labels'].to(device),
                }

                out_2 = m2(**inputs_token)
                loss = out_2.los
                eval_loss += loss.item()

                preds_list.append(out_2['logits'].detach().cpu().numpy())
                golds_list.append(inputs_token['labels'].detach().cpu().numpy())

                val_loss_tmp = round(eval_loss / (i + 1), 4)
                progbar_val.set_postfix({'Eval loss': val_loss_tmp})

        # Convert lists to numpy arrays after collecting all the data
        preds = np.vstack(preds_list)
        golds = np.vstack(golds_list)

        results = compute_metrics(preds, golds, label_list)

        results['json_path'] = json_path
        results['path_m1'] = path_m1
        results['path_m2'] = path_m2
        results['len(binary_dataset)'] = len(binary_dataset)
        results['json_path'] = json_path
        results['target_tag'] = tt[1]

        with open(os.path.join(path_m2, f'results_{tt[1]}.json'), 'w', encoding='utf8') as f:
            json.dump(dict(results), f, ensure_ascii = False)

if __name__ == "__main__":
    main()