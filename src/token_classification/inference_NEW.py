import logging
import json
import sys
sys.path.append('./src')
from train_sent import span_to_words_annotation, dict_of_lists, list_of_dicts, tokenize_token_classification
from seq_classification.train_seq import tokenize_sequence_classification
from utils_checkthat import regex_tokenizer_mappings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, DataCollatorWithPadding, DataCollatorForTokenClassification
import pandas as pd
from labelset import LabelSet
from datetime import datetime
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
    
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
SEQ_COLUMNS = ['lang', 'annotations', 'text', 'article_id', 'label', 'sent_start']
TOKEN_COLUMNS = ['id', 'ner_tags', 'tokens']
EXCLUSION_COLUMNS = ['data', 'annotations'] + SEQ_COLUMNS + TOKEN_COLUMNS
EXCLUSION_COLUMNS.remove('label')

def find_best_m2_checkpoint(model_dir, model_idx):
    """
    Find the best checkpoint for a given model index.
    """
    logging.info(f"Searching for best checkpoint for model index {model_idx}")
    for model_dir_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_dir_name)
        if os.path.isdir(model_path) and not model_dir_name.startswith('.'):
            try:
                current_model_number = int(model_dir_name.split('_')[2])
                if current_model_number == model_idx:
                    checkpoints = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)) and 'checkpoint' in d]
                    if len(checkpoints) >= 2:
                        second_checkpoint = checkpoints[1]
                        second_checkpoint_path = os.path.join(model_path, second_checkpoint)
                        trainer_state_file = os.path.join(second_checkpoint_path, 'trainer_state.json')
                        if os.path.exists(trainer_state_file):
                            with open(trainer_state_file, 'r') as f:
                                path_m2_tmp = json.load(f)['best_model_checkpoint']
                                split_path = path_m2_tmp.split("cw_ts0.9/")
                                best_model_checkpoint = split_path[0] + "cw_ts0.9/weights/" + split_path[1]                                
                                logging.info(f"Found best checkpoint: {best_model_checkpoint}")
                                return best_model_checkpoint
            except (IndexError, ValueError) as e:
                logging.warning(f"Skipping directory {model_dir_name}: {str(e)}")
    
    logging.error(f"No valid checkpoint found for model index {model_idx}")
    raise FileNotFoundError(f"No valid checkpoint found for model index {model_idx}")

def convert_tensors_for_submission(outputs_1, outputs_2, batch_1, batch_2, tokenizer, target_tag):
    preds_formatted = []
    predictions_2 = torch.argmax(outputs_2.logits, dim=2)
    
    # Instead of setting to 0, we'll use a threshold
    threshold = 0.5
    zero_indices = (torch.sigmoid(outputs_1.logits)[:, 1] < threshold).nonzero(as_tuple=True)[0] # get indices of 0 preds from sequence classifier
    predictions_2[zero_indices, :] = 0 # turn the same indexes into all 0s in the token predictions

    # convert the prediction tensor to the submission format
    token_list = batch_2['tokens']
    encodings = tokenizer(token_list, return_tensors='pt', padding='longest', truncation=True, is_split_into_words=True)

    total_potential_preds = 0
    total_actual_preds = 0

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
                total_potential_preds += 1
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
                    total_actual_preds += 1
                    span_left_idx_list = []
                    span_right_idx_list = []

    logging.info(f"Total potential predictions: {total_potential_preds}")
    logging.info(f"Total actual predictions: {total_actual_preds}")
    return preds_formatted

def main():
    try:
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        json_path = '/home/lgiordano/LUCA/checkthat_GITHUB/data/formatted/train_sentences.json'
        json_path_simple = json_path.split('/')[-1].split('.')[0]

        # Load and validate input data
        try:
            with open(json_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            if not data:
                raise ValueError("Input data is empty")
            logging.info(f"Successfully loaded {len(data)} samples from {json_path}")
        except Exception as e:
            logging.error(f"Error loading input data: {str(e)}")
            return

        langs = set([el['data']['lang'] for el in data])
        for lang in langs:
            logging.info(f'Inferring on {lang}...')

            path_m1 = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M1/RUN_OTTOBRE/aug/aug, lr 2e-5, + ARAIEVAL(news) & SEMEVAL24/checkpoint-27118' #'/home/lgiordano/LUCA/checkthat_GITHUB/models/M1/RUN_OTTOBRE/no aug, lr 5e-5/checkpoint-12507'
            model_dir_m2 = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M2/RUN_OTTOBRE/weights_and_results/2024-10-24-13-02-35_aug_cw_ts0.9/weights' #'/home/lgiordano/LUCA/checkthat_GITHUB/models/M2/RUN_OTTOBRE/weights_and_results/2024-10-31-10-16-40_aug_cw_ts0.9/weights'
            preds_dir = os.path.join('/home/lgiordano/LUCA/checkthat_GITHUB/preds/', f'DEV_{date_time}'+'_aug_cw_ts0.9_BEST_ARAIEVAL_SEMEVAL')
            os.makedirs(preds_dir, exist_ok=True)

            # Load and verify models
            try:
                tokenizer_m1 = AutoTokenizer.from_pretrained(path_m1)
                m1 = AutoModelForSequenceClassification.from_pretrained(path_m1)
                logging.info(f"Successfully loaded M1 model from {path_m1}")
            except Exception as e:
                logging.error(f"Error loading M1 model: {str(e)}")
                return

            data_lang = [el for el in data if el['data']['lang'] == lang]
            logging.info(f"Processing {len(data_lang)} samples for language {lang}")

            df_seq = pd.DataFrame(data_lang)
            dataset_seq = Dataset.from_pandas(df_seq)
            
            binary_dataset_seq = dataset_seq.map(
                lambda x: tokenize_sequence_classification(x, tokenizer_m1),
                batched=True,
                batch_size=None,
            )
            logging.info(f"Processed {len(binary_dataset_seq)} samples for sequence classification")

            target_tags = ["Appeal_to_Authority", "Appeal_to_Popularity", "Appeal_to_Values", "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
                        "False_Dilemma-No_Choice", "Consequential_Oversimplification", "Straw_Man", "Red_Herring", "Whataboutism", "Slogans", "Appeal_to_Time",
                        "Conversation_Killer", "Loaded_Language", "Repetition", "Exaggeration-Minimisation", "Obfuscation-Vagueness-Confusion", "Name_Calling-Labeling",
                        "Doubt", "Guilt_by_Association", "Appeal_to_Hypocrisy", "Questioning_the_Reputation"]
            target_tags = [(i, el.strip()) for i, el in enumerate(target_tags)]
            all_preds_formatted = []

            for model_idx, tt in enumerate(target_tags, start=0):
                logging.info(f'Inferring with m2 no. {model_idx} of {len(target_tags)-1} for {tt[1]} persuasion technique...')
                labels_model = LabelSet(labels=[tt[1]])
                
                df_list_binary = span_to_words_annotation(dict_of_lists(data_lang), target_tag=tt[1], mappings=regex_tokenizer_mappings, labels_model=labels_model)
                df_binary = pd.DataFrame(df_list_binary)
                binary_dataset_token = Dataset.from_pandas(df_binary[TOKEN_COLUMNS])
                
                # Load M2 model
                try:
                    path_m2 = find_best_m2_checkpoint(model_dir_m2, model_idx)
                    tokenizer_m2 = AutoTokenizer.from_pretrained(path_m2)
                    m2 = AutoModelForTokenClassification.from_pretrained(path_m2,
                                                                        num_labels=len(labels_model.ids_to_label.values()),
                                                                        label2id=labels_model.labels_to_id,
                                                                        id2label=labels_model.ids_to_label)
                    logging.info(f"Successfully loaded M2 model from {path_m2}")
                except Exception as e:
                    logging.error(f"Error loading M2 model: {str(e)}")
                    continue

                binary_dataset_token = binary_dataset_token.map(
                    lambda x: tokenize_token_classification(x, tokenizer_m2),
                    batched=True,
                    batch_size=None,
                )
                logging.info(f"Processed {len(binary_dataset_token)} samples for token classification")

                # Set up data loaders
                val_loader_seq = DataLoader(binary_dataset_seq, batch_size=16, shuffle=False, collate_fn=SequenceCollator(tokenizer_m1, 'longest', EXCLUSION_COLUMNS))
                val_loader_token = DataLoader(binary_dataset_token, batch_size=16, shuffle=False, collate_fn=TokenCollator(tokenizer_m1, 'longest', EXCLUSION_COLUMNS))

                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                m1.to(device)
                m1.eval()
                m2.to(device)
                m2.eval()

                preds_formatted = []
                progbar_val = tqdm(zip(val_loader_seq, val_loader_token), total=len(val_loader_seq), desc='Inference...')
                for i, (batch_seq, batch_token) in enumerate(progbar_val):
                    try:
                        with torch.no_grad():
                            inputs_seq = {k: v.to(device) for k, v in batch_seq.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                            out_1 = m1(**inputs_seq)

                            inputs_token = {k: v.to(device) for k, v in batch_token.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                            out_2 = m2(**inputs_token)
                            
                            batch_preds = convert_tensors_for_submission(out_1, out_2, batch_seq, batch_token, tokenizer_m2, tt[1])
                            preds_formatted.extend(batch_preds)
                    except Exception as e:
                        logging.error(f"Error during inference on batch {i}: {str(e)}")

                logging.info(f"Generated {len(preds_formatted)} predictions for {tt[1]}")
                all_preds_formatted.extend(preds_formatted)

                # Write predictions to file
                output_file = os.path.join(preds_dir, f'{model_idx}_preds_{tt[1]}_{json_path_simple}_{lang}.txt')
                with open(output_file, 'w', encoding='utf8') as f:
                    for pred in preds_formatted:
                        f.write(pred + '\n')
                logging.info(f"Wrote predictions to {output_file}")

            # Write all predictions to a single file
            all_output_path = os.path.join(preds_dir, 'all')
            os.makedirs(all_output_path, exist_ok=True)
            all_output_file = os.path.join(all_output_path, f'all_preds_{json_path_simple}_{lang}.txt')
            with open(all_output_file, 'w', encoding='utf8') as f:
                for pred in all_preds_formatted:
                    f.write(pred + '\n')
            logging.info(f"Wrote all predictions to {all_output_file}")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    main()