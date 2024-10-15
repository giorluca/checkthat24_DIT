import json
import sys
sys.path.append('./src')
from train_sent import span_to_words_annotation, dict_of_lists, list_of_dicts, tokenize_token_classification
from seq_classification.train_seq import tokenize_sequence_classification
from utils_checkthat import regex_tokenizer_mappings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, DataCollatorWithPadding, DataCollatorForTokenClassification
import pandas as pd
from labelset import LabelSet
from datasets import Dataset
import os
from torch.utils.data import DataLoader
import torch
torch.set_printoptions(linewidth=100000, threshold=100000)
from tqdm import tqdm
import re

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

def convert_tensors_for_submission(outputs_1, outputs_2, batch_1, batch_2, tokenizer, target_tag):
    preds_formatted = []
    predictions_1 = torch.argmax(outputs_1.logits, dim=1)
    predictions_2 = torch.argmax(outputs_2.logits, dim=2)
    zero_indices = (predictions_1 == 0).nonzero(as_tuple=True)[0] # get indices of 0 preds from sequence classifier
    predictions_2[zero_indices, :] = 0 # turn the same indexes into all 0s in the token predictions
    
    # convert the prediction tensor to the submission format
    token_list = batch_2['tokens']
    encodings = tokenizer(token_list, return_tensors='pt', padding='longest', truncation=True, is_split_into_words=True)

    for j in range(encodings.input_ids.shape[0]):
        indeces = [int(el) for el in (predictions_2[j, :] != 0)] # binarized predictions
        word_list = []
        for idx, value in enumerate(indeces):
            word_idx = encodings.token_to_word(j, idx)
            word = batch_2['tokens'][j][word_idx] if word_idx is not None else None
            word_list.append({'idx': word_idx, 'value': value, 'word': word})
        span_left_idx_list = []
        span_right_idx_list = []
        shift = 0
        for k, w in enumerate(word_list):
            if w['word'] is not None:
                if w['value'] == 1:
                    span_left_idx = batch_1['text'][j][shift:].find(w['word'])
                    left = span_left_idx + shift + batch_1['sent_start'][j]
                    right = left + len(w['word'])
                    span_left_idx_list.append(left)
                    span_right_idx_list.append(right)
                elif k > 0 and span_left_idx_list and span_right_idx_list:
                    left = min(span_left_idx_list)
                    right = max(span_right_idx_list)
                    article_id = batch_1['article_id'][j]
                    preds_formatted.append(f'{article_id}\t{target_tag}\t{left}\t{right}')
                    span_left_idx_list = []
                    span_right_idx_list = []
    return preds_formatted

def main():

    json_path = '/home/lgiordano/LUCA/checkthat_GITHUB/data/formatted/test_sentences.json'
    json_path_simple = json_path.split('/')[-1].split('.')[0]

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    langs = set([el['data']['lang'] for el in data])
    for lang in langs:
        print(f'Infering on {lang}...')

        path_m1 = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M1/RUN_OTTOBRE/2nd_run/mdeberta-v3-base-NEW_2nd/checkpoint-8338'
        model_dir_m2 = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M2/RUN_OTTOBRE/weights_and_results/model_weights'
        preds_dir = os.path.join('/home/lgiordano/LUCA/checkthat_GITHUB/preds/', '2024-10-15-09-48-54_aug_cw_ts0.9')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)

        tokenizer_m1 = AutoTokenizer.from_pretrained(path_m1)

        data_lang = [el for el in data if el['data']['lang'] == lang]

        df_seq = pd.DataFrame(data_lang)
        seq_columns = ['lang', 'annotations', 'text', 'article_id', 'label', 'sent_start']
        dataset_seq = Dataset.from_pandas(df_seq)
        batch_size = 16
        binary_dataset_seq = dataset_seq.map(lambda x: tokenize_sequence_classification(x, tokenizer_m1),
                                                batched=True,
                                                batch_size=None,
                                                )
        
        target_tags = ["Appeal_to_Authority", "Appeal_to_Popularity","Appeal_to_Values","Appeal_to_Fear-Prejudice","Flag_Waving","Causal_Oversimplification",
               "False_Dilemma-No_Choice","Consequential_Oversimplification","Straw_Man","Red_Herring","Whataboutism","Slogans","Appeal_to_Time",
               "Conversation_Killer","Loaded_Language","Repetition","Exaggeration-Minimisation","Obfuscation-Vagueness-Confusion","Name_Calling-Labeling",
               "Doubt","Guilt_by_Association","Appeal_to_Hypocrisy","Questioning_the_Reputation"]
        target_tags = [(i, el.strip()) for i, el in enumerate(target_tags)]
        all_preds_formatted = []
        model_shift = 0
        for model_idx, tt in enumerate(target_tags):
            # if model_idx < model_shift:
            #     continue
            print(f'Infering with m2 no. {model_idx} of {len(target_tags)} for {tt[1]} persuasion technique...')
            labels_model = LabelSet(labels=[tt[1]])
            
            df_list_binary = span_to_words_annotation(dict_of_lists(data_lang), target_tag=tt[1], mappings=regex_tokenizer_mappings, labels_model=labels_model)
            df_binary = pd.DataFrame(df_list_binary)
            token_columns = ['id', 'ner_tags', 'tokens']
            binary_dataset_token = Dataset.from_pandas(df_binary[token_columns])
            
            for model_dir in os.listdir(model_dir_m2):
                if os.path.isdir(os.path.join(model_dir_m2, model_dir)) and not re.search(r'\..*$', model_dir):
                    model_number = int(model_dir.split('_')[2])
                    if model_number == model_idx:
                        model_path = os.path.join(model_dir_m2, model_dir)
                        checkpoints = sorted([d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)) and 'checkpoint' in d])
                        if len(checkpoints) >= 2:
                            second_checkpoint = checkpoints[1]  # Access the second checkpoint
                            second_checkpoint_path = os.path.join(model_path, second_checkpoint)
                            trainer_state_file = os.path.join(second_checkpoint_path, 'trainer_state.json')
                            if os.path.exists(trainer_state_file):
                                with open(trainer_state_file, 'r') as f:
                                    path_m2_tmp = json.load(f)['best_model_checkpoint']
                                    split_path = path_m2_tmp.split("weights_and_results/")
                                    path_m2 = split_path[0] + "weights_and_results/model_weights/" + split_path[1]
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
            preds_list = []
            golds_list = []
            # label_list = [value for value in labels_model.ids_to_label.values()]

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

                    inputs_token = {
                        'input_ids': batch_token['input_ids'].to(device),
                        'token_type_ids': batch_token['token_type_ids'].to(device),
                        'attention_mask': batch_token['attention_mask'].to(device),
                        # 'labels': batch_token['labels'].to(device),
                    }

                    out_2 = m2(**inputs_token)
                    
                    preds_list.append(out_2['logits'].detach().cpu().numpy())
                    if 'labels' in inputs_token.keys():
                        golds_list.append(inputs_token['labels'].detach().cpu().numpy())

                    val_loss_tmp = round(eval_loss / (i + 1), 4)
                    progbar_val.set_postfix({'Eval loss': val_loss_tmp})

                preds_formatted = convert_tensors_for_submission(out_1, out_2, batch_seq, batch_token, tokenizer_m2, tt[1])
            all_preds_formatted += preds_formatted
            with open(os.path.join(preds_dir, f'{model_idx}_preds_{tt[1]}_{json_path_simple}_{lang}.txt'), 'w', encoding='utf8') as f:
                for pred in preds_formatted:
                    f.write(pred + '\n')

            # # Convert lists to numpy arrays after collecting all the data
            # preds = np.vstack(preds_list)
            # golds = np.vstack(golds_list)

            # results = compute_metrics(preds, golds, label_list)

            # results['json_path'] = json_path
            # results['path_m1'] = path_m1
            # results['path_m2'] = path_m2
            # results['len(binary_dataset_seq)'] = len(binary_dataset_seq)
            # results['json_path'] = json_path
            # results['target_tag'] = tt[1]

            # with open(os.path.join(path_m2, f'results_{tt[1]}_{lang}.json'), 'w', encoding='utf8') as f:
            #     json.dump(dict(results), f, ensure_ascii = False)
        all_output_path = os.path.join(preds_dir, 'all')
        if not os.path.exists(all_output_path):
            os.makedirs(all_output_path)
        with open(os.path.join(all_output_path, f'all_preds_{json_path_simple}_{lang}.txt'), 'w', encoding='utf8') as f:
            for pred in all_preds_formatted:
                f.write(pred + '\n')

if __name__ == "__main__":
    main()