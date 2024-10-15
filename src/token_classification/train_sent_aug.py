import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from datasets import Dataset, concatenate_datasets
import pandas as pd
import os
import sys
sys.path.append('./src')
from checkthat_GITHUB.src.token_classification.labelset import LabelSet
from utils_checkthat import save_local_model, sub_shift_spans, regex_tokenizer_mappings, get_entities_from_sample
from icecream import ic
import re
from datetime import datetime
from collections import defaultdict
import evaluate
from sklearn.metrics import f1_score, precision_score
from collections import Counter
from seqeval.metrics import classification_report
from transformers import DataCollatorForTokenClassification

from torch.nn import CrossEntropyLoss

class WeightedLoss(CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(WeightedLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):
        # Ensure weight tensor is on the same device as input
        weight = torch.tensor([0.5 if x == 0 else 2.0 for x in target.view(-1)], device=input.device, dtype=input.dtype)
        # Ensure target is on the same device as input
        target = target.to(input.device)
        loss = super(WeightedLoss, self).forward(input.view(-1, input.size(-1)), target.view(-1))
        return (loss * weight).mean()

def tokenize_token_classification(examples, tokenizer):
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

def list_of_dicts(dict_of_lists):
    # First, we need to check if all lists are of the same length to ensure correct transformation
    if not all(len(lst) == len(next(iter(dict_of_lists.values()))) for lst in dict_of_lists.values()):
        raise ValueError("All lists in the dictionary must have the same length")

    # Get the length of the items in any of the lists
    length = len(next(iter(dict_of_lists.values())))
    
    # Create a list of dictionaries, one for each index in the lists
    result = []
    for i in range(length):
        # Create a dictionary for the current index 'i' across all lists
        new_dict = {key: dict_of_lists[key][i] for key in dict_of_lists}
        result.append(new_dict)
    
    return result

def span_to_words_annotation(samples, target_tag = '', mappings = {}, labels_model = []):
    samples_new = []
    # if not any([l for l in samples['annotations']]):
        
    for i in range(len(samples['data'])):
        text, annotation_list = samples['data'][i]['text'], samples['annotations'][i][0]['result']
        labels_text = []
        tokens = []
        if not annotation_list:
            annotation_list = [[]]
        for j, annotation in enumerate(annotation_list):
            if isinstance(annotation, dict):
                if annotation['value']['labels'][0] != target_tag:
                    continue
            text_subshifted, ents = sub_shift_spans(text, annotation, mappings=mappings)
            text_subshifted_matches = re.finditer(r'[^\s]+', text_subshifted)
            labels_words = []
            first = True
            for regex_match in text_subshifted_matches:
                if j == 0:
                    tokens.append(regex_match.group())
                if isinstance(annotation, dict):
                    if regex_match.start() < ents['value']['start']:
                        labels_words.append(labels_model.labels_to_id['O'])
                    elif regex_match.start() >= ents['value']['start'] and regex_match.end() <= ents['value']['end']:
                        if first:
                            labels_words.append(labels_model.labels_to_id['B-' + ents['value']['labels'][0]])
                            first = False
                        elif not first:
                            labels_words.append(labels_model.labels_to_id['I-' + ents['value']['labels'][0]])
                    else:
                        labels_words.append(labels_model.labels_to_id['O'])
                    labels_text.append({'labels': labels_words, 'tag': annotation['value']['labels'][0]})
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
            'lang': samples['data'][i]['lang'],
        })
    return samples_new

def compute_metrics(predictions, labels, label_list, threshold=0.9):
    # Apply softmax to predictions to get probabilities
    predictions = torch.softmax(torch.tensor(predictions), dim=-1).numpy()

    # Apply threshold to make predictions more cautious
    predictions = np.where(predictions >= threshold, predictions, 0)
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

    # Calculate precision and micro F1-score using sklearn
    precision = precision_score(flat_true_labels, flat_true_predictions, average='micro')
    micro_f1 = f1_score(flat_true_labels, flat_true_predictions, average='micro')

    # Prepare the results dictionary
    flat_results = {
        'micro_f1': float(micro_f1),
        'precision': float(precision)
    }

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

def evaluate(model, val_loader, label_list, device, threshold=0.9):
    model.eval()
    eval_loss = 0
    preds_list = []
    golds_list = []

    loss_fn = WeightedLoss()

    progbar_val = tqdm(val_loader)
    for i, batch in enumerate(progbar_val):
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None,
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device),
            }

            # Remove None values in inputs
            inputs = {k: v for k, v in inputs.items() if v is not None}

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits.view(-1, model.config.num_labels), inputs['labels'].view(-1))
            eval_loss += loss.item()

            preds_list.append(outputs.logits.detach().cpu().numpy())
            golds_list.append(inputs['labels'].detach().cpu().numpy())

            val_loss_tmp = round(eval_loss / (i + 1), 4)
            progbar_val.set_postfix({'Eval loss': val_loss_tmp})

    # Convert lists to numpy arrays after collecting all the data
    preds = np.vstack(preds_list)
    golds = np.vstack(golds_list)

    results = compute_metrics(preds, golds, label_list, threshold)

    return results

def make_balanced_df(df):
    # get rows with annotations
    df_pos = df[df['annotations'].apply(lambda x: len(x[0]['result']) > 0)]
    # get the same number of rows without any annotations
    df_neg = df[df['annotations'].apply(lambda x: x[0]['result'] == [])].sample(len(df_pos))
    balanced_df = pd.concat([df_pos, df_neg])
    return balanced_df

def make_binary_balanced_df(df, target_tag='', labels_model=[]):
    df_list = df.to_dict(orient='records')
    df_list_binary = span_to_words_annotation(dict_of_lists(df_list), target_tag=target_tag, mappings=regex_tokenizer_mappings, labels_model=labels_model)
    df_binary = pd.DataFrame(df_list_binary)
    df_binary_pos = df_binary[df_binary['tag'] == target_tag]
    df_binary_neg = df_binary[df_binary['tag'] != target_tag].sample(len(df_binary_pos), replace=True)  # Over-sampling
    df_binary_subsampled = pd.concat([df_binary_pos, df_binary_neg])
    return df_binary_subsampled

def main():
    data_gold = './data/formatted/train_sentences.json'
    with open(data_gold, 'r', encoding='utf8') as f:
        dataset_gold = json.load(f)

    data_path_dict = {
    'sl': './data/train_sent_mt/sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl_mdeberta-v3-base_mdeberta_xlwa_en-sl_ME3_2024-05-04-12-12-14_ls.json',
    'ru': './data/train_sent_mt/ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru_mdeberta-v3-base_mdeberta_xlwa_en-ru_ME3_2024-05-04-12-09-20_ls.json',
    'pt': './data/train_sent_mt/pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt_mdeberta-v3-base_mdeberta_xlwa_en-pt_ME3_2024-05-04-12-07-45_ls.json',
    'it': './data/train_sent_mt/it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it_mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-05-04-12-05-00_ls.json',
    'es': './data/train_sent_mt/es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es_mdeberta-v3-base_mdeberta_xlwa_en-es_ME3_2024-05-04-12-01-43_ls.json',
    'bg': './data/train_sent_mt/bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg_mdeberta-v3-base_mdeberta_xlwa_en-bg_ME3_2024-05-04-11-58-52_ls.json',
    }
    dataset_aug = []
    for key in data_path_dict:
        with open(data_path_dict[key], 'r', encoding='utf8') as f:
            dataset_aug_buffer = json.load(f)
            for sample in dataset_aug_buffer:
                sample['annotations'][0]['result'] = get_entities_from_sample(sample, langs=[key], sort = True)
                del sample['data']['text_en']
                sample['data']['text'] = sample['data'][f'text_{key}']
                del sample['data'][f'text_{key}']
                sample['data']['lang'] = key
            dataset_aug += dataset_aug_buffer

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    df_gold = pd.DataFrame(dataset_gold)
    balanced_df_gold = make_balanced_df(df_gold)

    df_aug = pd.DataFrame(dataset_aug)
    balanced_df_aug = make_balanced_df(df_aug)
    
    target_tags = [(i, el.strip()) for i, el in enumerate(open('./data/persuasion_techniques.txt').readlines())]
    shift = 14
    for i, tt in enumerate(target_tags):
        if i < shift:
            continue
        print(f'Training model no. {i} of {len(target_tags)} for {tt} persuasion technique...')
        labels_model = LabelSet(labels=[tt[1]])
        
        token_columns = ['id', 'ner_tags', 'tokens', 'lang']

        df_binary_subsampled_gold = make_binary_balanced_df(balanced_df_gold, target_tag=tt[1], labels_model=labels_model)
        binary_dataset_gold = Dataset.from_pandas(df_binary_subsampled_gold[token_columns])

        df_binary_subsampled_aug = make_binary_balanced_df(balanced_df_aug, target_tag=tt[1], labels_model=labels_model)
        binary_dataset_aug = Dataset.from_pandas(df_binary_subsampled_aug[token_columns])
        
        split_ratio = 0.2
        split_seed = 42
        datadict = binary_dataset_gold.train_test_split(split_ratio, seed=split_seed)

        # model_name = 'bert-base-multilingual-cased'
        # model_name = 'xlm-roberta-base'
        model_name = 'microsoft/mdeberta-v3-base'
        # model_name = 'FacebookAI/xlm-roberta-large'
        model_name_simple = model_name.split('/')[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 16
        ic(datadict)
        datadict['train'] = concatenate_datasets([datadict['train'], binary_dataset_aug]) # this is where we merge english gold data with aug data
        ic(datadict)
        datadict = datadict.map(lambda x: tokenize_token_classification(x, tokenizer), batched=True, batch_size=None)

        columns = [
                    'input_ids',
                    'token_type_ids',
                    'attention_mask',
                    'labels'
                    ]

        datadict.set_format('torch', columns = columns)

        train_data = datadict['train']#.select(range(100))
        val_data = datadict['test']

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding = 'longest')

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                                num_labels=len(labels_model.ids_to_label.values()),
                                                                label2id=labels_model.labels_to_id,
                                                                id2label=labels_model.ids_to_label,
                                                                )
        lr = 5e-5
        optimizer = AdamW(model.parameters(), lr=lr)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        epochs = 10

        model.to(device)

        patience = 2
        patience_counter = 0
        best_f1_score = 0
        best_model = model
        best_epoch = -1
        results = {'results': []}
        
        loss_fn = WeightedLoss()
        threshold = 0.9
        for epoch in range(epochs):
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

                # loss = outputs.loss
                loss = loss_fn(outputs.logits.view(-1, model.config.num_labels), batch['labels'].view(-1))
                loss.backward()
                train_loss += loss.item()
                
                optimizer.step()

                train_loss_tmp = round(train_loss / (i + 1), 4)
                progbar_train.set_postfix({'Train loss':train_loss_tmp})

            results_entry = evaluate(model, val_loader, [value for value in labels_model.ids_to_label.values()], device, threshold = threshold)
            results_entry['epoch'] = epoch
            results['results'].append(results_entry)
            print(results['results'][-1])
            print(results)

            current_f1_score = results['results'][-1]['macro avg_f1-score']
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_model = model
                best_epoch = epoch
                patience_counter = 0
                print(f'Best model updated: current epoch macro f1 = {current_f1_score}')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        results['best_epoch'] = best_epoch + 1
        
        models_dir = './models/M2'
        model_save_name = f'{model_name_simple}_{tt[0]}_ME{epochs}_target={tt[1]}_SUBSAMPLED_{date_time}'
        model_save_dir = os.path.join(models_dir, date_time+'_aug'+'_ts'+str(threshold), model_save_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        save_local_model(model_save_dir, best_model, tokenizer)

        results['train_data_path'] = data_path_dict  
        ic(type(results))
        ic(results)

        with open(os.path.join(model_save_dir, 'results.json'), 'w', encoding='utf8') as f:
            json.dump(dict(results), f, ensure_ascii = False)

        info = {
            'model_name': model_name,
            'epochs': epochs,
            'lr': lr,
            'len(train_data)': len(train_data),
            'len(val_data)': len(val_data),
            'split_ratio': split_ratio,
            'split_seed': split_seed,
            'train_data_path': data_path_dict,
        }

        with open(os.path.join(model_save_dir, 'info.json'), 'w', encoding='utf8') as f:
            json.dump(info, f, ensure_ascii = False)

if __name__ == "__main__":
    main()