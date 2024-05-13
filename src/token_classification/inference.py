import json
import sys
sys.path.append('./src')
from token_classification.train_sent import span_to_words_annotation, dict_of_lists, list_of_dicts, tokenize_token_classification, compute_metrics
from seq_classification.train_seq import tokenize_sequence_classification
from utils_checkthat import regex_tokenizer_mappings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, DataCollatorWithPadding, DataCollatorForTokenClassification
import pandas as pd
from sequence_aligner.labelset import LabelSet
from datasets import Dataset
import os
from torch.utils.data import DataLoader
import torch
torch.set_printoptions(linewidth=100000, threshold=100000)
from tqdm import tqdm
import re
import numpy as np

class SequenceCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding, exclusion_columns):
        self.exclusion_columns = exclusion_columns
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features):
        features = dict_of_lists(features)
        raw_columns = {}
        for column in self.exclusion_columns:
            if column in features.keys():
                raw_columns[column] = features[column]
                del features[column]
        output_batch = super().__call__(features)
        output_batch.update(raw_columns)
        return output_batch

class TokenCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, padding, exclusion_columns):
        self.exclusion_columns = exclusion_columns
        self.tokenizer = tokenizer
        self.padding = padding

    def torch_call(self, features):
        features = dict_of_lists(features)
        raw_columns = {}
        for column in self.exclusion_columns:
            if column in features.keys():
                raw_columns[column] = features[column]
                del features[column]
        output_batch = super().torch_call(list_of_dicts(features))
        output_batch.update(raw_columns)
        return output_batch

def main():

    json_path = './data/formatted/dev_sentences.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    path_m1 = './models/M1/mdeberta-v3-base_ME3_2024-05-08-20-23-31'

    model_dir_m2 = './models/M2/2024-05-12-17-42-10_aug'
    preds_dir = os.path.join('./preds/', model_dir_m2.split('/')[-1])
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)

    tokenizer_m1 = AutoTokenizer.from_pretrained(path_m1)

    df_seq = pd.DataFrame(data)
    seq_columns = ['lang', 'annotations', 'text', 'article_id', 'label', 'sent_start']
    dataset_seq = Dataset.from_pandas(df_seq)
    batch_size = 16
    binary_dataset_seq = dataset_seq.map(lambda x: tokenize_sequence_classification(x, tokenizer_m1),
                                            batched=True,
                                            batch_size=None,
                                            )
    
    target_tags = [(i, el.strip()) for i, el in enumerate(open('./data/persuasion_techniques.txt').readlines())]
    all_preds_formatted = []
    model_shift = 2
    for model_idx, tt in enumerate(target_tags):
        # if model_idx < model_shift:
        #     continue
        print(f'Infering with m2 no. {model_idx} of {len(target_tags)} for {tt} persuasion technique...')
        labels_model = LabelSet(labels=[tt[1]])
        
        df_list_binary = span_to_words_annotation(dict_of_lists(data), target_tag=tt[1], mappings=regex_tokenizer_mappings, labels_model=labels_model)
        df_binary = pd.DataFrame(df_list_binary)
        token_columns = ['id', 'ner_tags', 'tokens']
        binary_dataset_token = Dataset.from_pandas(df_binary[token_columns])
        
        for model_dir in os.listdir(model_dir_m2):
            if not re.search(r'\..*$', model_dir):
                model_number = int(model_dir.split('_')[1])
                if model_number == model_idx:
                    path_m2 = os.path.join(model_dir_m2, model_dir)
                    break

        
        columns = [ 'input_ids',
            'token_type_ids',
            'attention_mask',
            'labels'
            ]

        exclusion_columns = ['data', 'annotations'] + seq_columns + token_columns
        exclusion_columns.remove('label')

        binary_dataset_seq.set_format('torch',
                                    #   columns = columns
                                    )
        
        data_collator_seq = SequenceCollator(tokenizer=tokenizer_m1,
                                                    padding='longest',
                                                    exclusion_columns=exclusion_columns
                                                    )

        batch_size = 16

        val_loader_seq = DataLoader(binary_dataset_seq,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=data_collator_seq
                                    )

        m2_simple = path_m2.split('/')[-1]
        tokenizer_m2 = AutoTokenizer.from_pretrained(path_m2)

        binary_dataset_token = binary_dataset_token.map(lambda x: tokenize_token_classification(x, tokenizer_m2),
                                                            batched=True,
                                                            batch_size=None,
                                                            )
        binary_dataset_token.set_format('torch',
                                        # columns = columns
                                        )
        data_collator_token = TokenCollator(tokenizer=tokenizer_m1,
                                                padding='longest',
                                                exclusion_columns=exclusion_columns
                                                )
        val_loader_token = DataLoader(binary_dataset_token,
                                        batch_size=16,
                                        shuffle=False,
                                        collate_fn=data_collator_token
                                        )
        
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
        preds_formatted = []
        preds_list = []
        golds_list = []
        label_list = [value for value in labels_model.ids_to_label.values()]

        progbar_val = tqdm(zip(val_loader_seq, val_loader_token), total=len(val_loader_seq), desc='Inference...')
        for i, (batch_seq, batch_token) in enumerate(progbar_val):
            with torch.no_grad():
                inputs_seq = {
                    'input_ids': batch_seq['input_ids'].to(device),
                    'token_type_ids': batch_seq['token_type_ids'].to(device),
                    'attention_mask': batch_seq['attention_mask'].to(device),
                    # 'labels': batch_seq['labels'].to(device),
                }

                out_1 = m1(**inputs_seq)

                preds_1 = torch.argmax(out_1.logits, dim=1)
                # print('preds_1', preds_1)

                inputs_token = {
                    'input_ids': batch_token['input_ids'].to(device),
                    'token_type_ids': batch_token['token_type_ids'].to(device),
                    'attention_mask': batch_token['attention_mask'].to(device),
                    # 'labels': batch_token['labels'].to(device),
                }

                out_2 = m2(**inputs_token)

                preds_2 = torch.argmax(out_2.logits, dim=2)
                # print('preds_2', preds_2)
                zero_indices = (preds_1 == 0).nonzero(as_tuple=True)[0] # get indices of 0 preds from sequence classifier
                preds_2[zero_indices, :] = 0 # turn the same indexes into all 0s in the token predictions
                print('preds_2', preds_2)
                
                # convert the prediction tensor to the submission format
                token_list = batch_token['tokens']
                encodings = tokenizer_m2(token_list, return_tensors='pt', padding='longest', truncation=True, is_split_into_words=True)

                for j in range(encodings.input_ids.shape[0]):
                    indeces = [int(el) for el in (preds_2[j, :] != 0)] # binarized predictions
                    word_list = []
                    for idx, value in enumerate(indeces):
                        word_idx = encodings.token_to_word(j, idx)
                        word = batch_token['tokens'][j][word_idx] if word_idx is not None else None
                        word_list.append({'idx': word_idx, 'value': value, 'word': word})
                    span_left_idx_list = []
                    span_right_idx_list = []
                    shift = 0
                    for k, w in enumerate(word_list):
                        if w['word'] is not None:
                            if w['value'] == 1:
                                span_left_idx = batch_seq['text'][j][shift:].find(w['word'])
                                left = span_left_idx + shift + batch_seq['sent_start'][j]
                                right = left + len(w['word'])
                                span_left_idx_list.append(left)
                                span_right_idx_list.append(right)
                            elif k > 0 and span_left_idx_list and span_right_idx_list:
                                left = min(span_left_idx_list)
                                right = max(span_right_idx_list)
                                article_id = batch_seq['article_id'][j]
                                preds_formatted.append(f'{article_id}\t{tt[1]}\t{left}\t{right}')
                                span_left_idx_list = []
                                span_right_idx_list = []
                
                preds_list.append(out_2['logits'].detach().cpu().numpy())
                if 'labels' in inputs_token.keys():
                    golds_list.append(inputs_token['labels'].detach().cpu().numpy())

                val_loss_tmp = round(eval_loss / (i + 1), 4)
                progbar_val.set_postfix({'Eval loss': val_loss_tmp})

        all_preds_formatted += preds_formatted
        with open(os.path.join(preds_dir, f'{model_idx}_preds_{tt[1]}.txt'), 'w', encoding='utf8') as f:
            for pred in preds_formatted:
                f.write(pred + '\n')

        # Convert lists to numpy arrays after collecting all the data
        preds = np.vstack(preds_list)
        golds = np.vstack(golds_list)

        results = compute_metrics(preds, golds, label_list)

        results['json_path'] = json_path
        results['path_m1'] = path_m1
        results['path_m2'] = path_m2
        results['len(binary_dataset_seq)'] = len(binary_dataset_seq)
        results['json_path'] = json_path
        results['target_tag'] = tt[1]

        with open(os.path.join(path_m2, f'results_{tt[1]}.json'), 'w', encoding='utf8') as f:
            json.dump(dict(results), f, ensure_ascii = False)

    with open(os.path.join(preds_dir, f'all_preds.txt'), 'w', encoding='utf8') as f:
        for pred in all_preds_formatted:
            f.write(pred + '\n')

if __name__ == "__main__":
    main()