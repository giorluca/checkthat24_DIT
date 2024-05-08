import json
from sequence_aligner.labelset import LabelSet
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from datasets import Dataset as Dataset
import pandas as pd
import os
import sys
sys.path.append('./src')
from utils_checkthat import save_local_model, sub_shift_spans
from icecream import ic
import re
from datetime import datetime
from collections import defaultdict
import evaluate
from sklearn.metrics import f1_score
from collections import Counter
from seqeval.metrics import classification_report

def tokenize_and_align_labels(examples, tokenizer):
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

def dict_of_lists(lst_of_dicts):
    result = defaultdict(list)
    for d in lst_of_dicts:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)

def span_to_words_annotation(samples, target_tag = '', mappings = {}, labels_model = []):
    samples_new = []
    for i, (text, annotation_list) in enumerate(zip(samples['text'], samples['annotations'])):
        labels_text = []
        tokens = []
        for j, annotation in enumerate(annotation_list):
            if annotation['tag'] != target_tag:
                continue
            text_subshifted, ents = sub_shift_spans(text, annotation, mappings=mappings)
            text_subshifted_matches = re.finditer(r'[^\s]+', text_subshifted)
            labels_words = []
            first = True
            for regex_match in text_subshifted_matches:
                if j == 0:
                    tokens.append(regex_match.group())
                if regex_match.start() < ents['start']:
                    labels_words.append(labels_model.labels_to_id['O'])
                elif regex_match.start() >= ents['start'] and regex_match.end() <= ents['end']:
                    if first:
                        labels_words.append(labels_model.labels_to_id['B-' + ents['tag']])
                        first = False
                    elif not first:
                        labels_words.append(labels_model.labels_to_id['I-' + ents['tag']])
                else:
                    labels_words.append(labels_model.labels_to_id['O'])
            labels_text.append({'labels': labels_words, 'tag': annotation['tag']})
        allowed_labels = [labels_model.labels_to_id['O'],
                          labels_model.labels_to_id['B-' + target_tag],
                          labels_model.labels_to_id['I-' + target_tag],
                          ]
        # if the training sample has no tags that we need, we just produce a 0s list
        if target_tag not in [labels['tag'] for labels in labels_text]:
            labels = [0] * len(tokens)
            tag = 'no_tag'
        # if the training sample has tags we need, we first exclude the label lists whose tags don't match
        # and then we merge the label lists that have tags that match the target tag
        else:
            labels = [max(values) for values in zip(*[labels['labels'] for labels in labels_text if labels['tag'] == target_tag])]
            labels = [(label if label in allowed_labels else 0) for label in labels]
            tag = target_tag
        samples_new.append({
            'id': i,
            'ner_tags': labels,
            'tokens': tokens,
            'tag': tag,
        })
    return samples_new

def compute_metrics(predictions, labels, label_list):
    predictions = np.argmax(predictions, axis=2)

    # Extract the true predictions and labels from the sequences
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute sequence-level evaluation metrics
    results = classification_report(true_predictions, true_labels, output_dict=True)

    # Flatten the lists to calculate micro F1-score and supports
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate micro F1-score using sklearn
    micro_f1 = f1_score(flat_true_labels, flat_true_predictions, average='micro')

    # Prepare the results dictionary
    flat_results = {'micro_f1': float(micro_f1)}

    # Add detailed metrics for each label to the results dictionary
    for label, metrics in results.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                flat_results[f'{label}_{metric}'] = float(value)

    # Compute support for each label using Counter
    label_support = Counter(flat_true_labels)
    for label, count in label_support.items():
        flat_results[f'{label}_support'] = count

    return flat_results

def evaluate(model, val_loader, label_list, device):
    model.eval()
    eval_loss = 0
    preds_list = []
    golds_list = []

    progbar_val = tqdm(val_loader)
    for i, batch in enumerate(progbar_val):
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device),
            }

            outputs = model(**inputs)
            loss = outputs.loss
            eval_loss += loss.item()

            preds_list.append(outputs['logits'].detach().cpu().numpy())
            golds_list.append(inputs['labels'].detach().cpu().numpy())

            val_loss_tmp = round(eval_loss / (i + 1), 4)
            progbar_val.set_postfix({'Eval loss': val_loss_tmp})

    # Convert lists to numpy arrays after collecting all the data
    preds = np.vstack(preds_list)
    golds = np.vstack(golds_list)

    results = compute_metrics(preds, golds, label_list)

    return results

def main():
    train_data_path = './data/formatted/train_sentences.json'
    with open(train_data_path, 'r', encoding='utf8') as f:
        dataset_raw = json.load(f)

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    df_raw = pd.DataFrame(dataset_raw)  

    df_pos = df_raw[df_raw['annotations'].apply(lambda x: len(x) > 0)]
    df_neg = df_raw[df_raw['annotations'].apply(lambda x: x == [])].sample(len(df_pos))
    df = pd.concat([
        df_pos,
        df_neg
        ]
        )
    
    mappings = [
        {'pattern': r'(?<!\s)([^\w\s])|([^\w\s])(?!\s)', 'target': ' placeholder '},
        {'pattern': r'\s+', 'target': ' '},
        ]
    
    target_tags = [(i, el.strip()) for i, el in enumerate(open('./data/persuasion_techniques.txt').readlines())]
    for i, tt in enumerate(target_tags):
        print(f'Training model no. {i} of {len(target_tags)} for {tt} persuasion technique...')
        labels_model = LabelSet(labels=[tt[1]])
        
        df_list = df.to_dict(orient='records')
        df_list_binary = span_to_words_annotation(dict_of_lists(df_list), target_tag=tt[1], mappings=mappings, labels_model=labels_model)
        df_binary = pd.DataFrame(df_list_binary)
        ic(df_binary['tag'].value_counts())
        df_binary_pos = df_binary[df_binary['tag'] == tt[1]]
        df_binary_neg = df_binary[df_binary['tag'] != tt[1]].sample(len(df_binary_pos))
        ic(df_binary_pos['tag'].value_counts())
        ic(df_binary_neg['tag'].value_counts())
        df_binary_subsampled = pd.concat([df_binary_pos, df_binary_neg])#.sample(1000)
        ic(df_binary_subsampled['tag'].value_counts())

        binary_dataset = Dataset.from_pandas(df_binary_subsampled[['id', 'ner_tags', 'tokens']])#.filter(lambda example: example["text"] is not None)
        # some samples have no text and cannot be tokenized so we filter them out
        
        split_ratio = 0.2
        split_seed = 42
        datadict = binary_dataset.train_test_split(split_ratio, seed=split_seed)

        # model_name = 'bert-base-multilingual-cased'
        # model_name = 'xlm-roberta-base'
        model_name = 'microsoft/mdeberta-v3-base'
        model_name_simple = model_name.split('/')[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        datadict = datadict.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, batch_size=None)

        columns = [
                    'input_ids',
                    'token_type_ids',
                    'attention_mask',
                    'labels'
                    ]

        datadict.set_format('torch', columns = columns)

        train_data = datadict['train']#.select(range(100))
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

        for _ in range(epochs):
            model.train()
            train_loss = 0
            progbar_train = tqdm(train_loader)
            for i, batch in enumerate(progbar_train):
                optimizer.zero_grad()
                inputs = {
                'input_ids': batch['input_ids'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
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

            final_results = evaluate(model, val_loader, [value for value in labels_model.ids_to_label.values()], device)
            print(final_results)

        models_dir = './models/M2'
        model_save_name = f'{model_name_simple}_{tt[0]}_ME{epochs}_target={tt[1]}_SUBSAMPLED_{date_time}'
        model_save_dir = os.path.join(models_dir, model_save_name)
        save_local_model(model_save_dir, model, tokenizer)

        final_results['train_data_path'] = train_data_path  
        ic(type(final_results))
        ic(final_results)

        with open(os.path.join(model_save_dir, 'results.json'), 'w', encoding='utf8') as f:
            json.dump(dict(final_results), f, ensure_ascii = False)

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