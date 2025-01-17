import logging
import json
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any
from labelset import LabelSet
import pandas as pd
import numpy as np
sys.path.append('./src')
from train_sent import span_to_words_annotation, dict_of_lists, list_of_dicts, tokenize_token_classification
from seq_classification.train_seq import tokenize_sequence_classification
from utils_checkthat import regex_tokenizer_mappings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, DataCollatorWithPadding, DataCollatorForTokenClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
torch.set_printoptions(linewidth=100000, threshold=100000)
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configuration
CONFIG = {
    'batch_size': 64,
    'chunk_size': 2000,
    'cleanup_interval': 10,
    'model_paths': {
        'm1': '/home/lgiordano/LUCA/checkthat_GITHUB/models/M1/RUN_OTTOBRE/no aug, lr 2e-5/2025-01-16-14-39-53/checkpoint-8338',
        'm2': '/home/lgiordano/LUCA/checkthat_GITHUB/models/M2/RUN_OTTOBRE/weights_and_results/2025-01-16-15-10-10_no_aug_no_cw_ts0/weights'
    },
    'target_tags': [
        "Appeal_to_Authority", "Appeal_to_Popularity", "Appeal_to_Values", 
        "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
        "False_Dilemma-No_Choice", "Consequential_Oversimplification", "Straw_Man", 
        "Red_Herring", "Whataboutism", "Slogans", "Appeal_to_Time",
        "Conversation_Killer", "Loaded_Language", "Repetition", 
        "Exaggeration-Minimisation", "Obfuscation-Vagueness-Confusion", 
        "Name_Calling-Labeling", "Doubt", "Guilt_by_Association", 
        "Appeal_to_Hypocrisy", "Questioning_the_Reputation"
    ],
    'aug' : [
        'no_aug',
        'aug_IT&RU',
        'aug',
        'aug+ARAIEVAL+SemEval'
    ]
}

# Constants
SEQ_COLUMNS = ['lang', 'annotations', 'text', 'article_id', 'label', 'sent_start']
TOKEN_COLUMNS = ['id', 'ner_tags', 'tokens']
EXCLUSION_COLUMNS = ['data', 'annotations', 'model'] + SEQ_COLUMNS + TOKEN_COLUMNS
EXCLUSION_COLUMNS.remove('label')

class MemoryEfficientDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        cleanup_interval = kwargs.pop('cleanup_interval', CONFIG['cleanup_interval'])
        super().__init__(*args, **kwargs)
        self.cleanup_interval = cleanup_interval
        self.current_batch = 0

    def __iter__(self):
        for batch in super().__iter__():
            self.current_batch += 1
            if self.current_batch % self.cleanup_interval == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield batch

class SequenceCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding, exclusion_columns):
        super().__init__(tokenizer=tokenizer, padding=padding)
        self.exclusion_columns = exclusion_columns
    
    def __call__(self, features):
        features_dict = self._convert_to_dict(features)
        raw_columns = self._extract_raw_columns(features_dict)
        output_batch = super().__call__(features_dict)
        output_batch.update(raw_columns)
        return output_batch
    
    def _convert_to_dict(self, features):
        if isinstance(features, dict):
            return features
        return {k: [d[k] for d in features] for k in features[0].keys()}
    
    def _extract_raw_columns(self, features):
        raw_columns = {}
        for column in self.exclusion_columns:
            if column in features:
                raw_columns[column] = features[column]
                del features[column]
        return raw_columns

class TokenCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, padding, exclusion_columns):
        super().__init__(tokenizer=tokenizer, padding=padding)
        self.exclusion_columns = exclusion_columns

    def torch_call(self, features):
        features_dict = self._convert_to_dict(features)
        raw_columns = self._extract_raw_columns(features_dict)
        features_list = [{k: v[i] for k, v in features_dict.items()} 
                        for i in range(len(next(iter(features_dict.values()))))]
        output_batch = super().torch_call(features_list)
        output_batch.update(raw_columns)
        return output_batch
    
    def _convert_to_dict(self, features):
        if isinstance(features, dict):
            return features
        return {k: [d[k] for d in features] for k in features[0].keys()}
    
    def _extract_raw_columns(self, features):
        raw_columns = {}
        for column in self.exclusion_columns:
            if column in features:
                raw_columns[column] = features[column]
                del features[column]
        return raw_columns

def setup_logging() -> None:
    """Set up logging configuration."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def find_best_m2_checkpoint(model_dir: str, model_idx: int) -> str:
    """Find the best checkpoint for a given model index."""
    logging.info(f"Searching for best checkpoint for model index {model_idx}")
    
    for model_dir_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_dir_name)
        if not os.path.isdir(model_path) or model_dir_name.startswith('.'):
            continue
            
        try:
            current_model_number = int(model_dir_name.split('_')[2])
            if current_model_number != model_idx:
                continue
                
            checkpoints = [d for d in os.listdir(model_path) 
                          if os.path.isdir(os.path.join(model_path, d)) and 'checkpoint' in d]
                
            if len(checkpoints) < 2:
                continue
                
            second_checkpoint = checkpoints[1]
            checkpoint_path = os.path.join(model_path, second_checkpoint)
            trainer_state_file = os.path.join(checkpoint_path, 'trainer_state.json')
            
            if not os.path.exists(trainer_state_file):
                continue
                
            with open(trainer_state_file, 'r') as f:
                best_model_checkpoint = json.load(f)['best_model_checkpoint']
                logging.info(f"Found best checkpoint: {best_model_checkpoint}")
                return best_model_checkpoint
                
        except (IndexError, ValueError, KeyError) as e:
            logging.warning(f"Error processing directory {model_dir_name}: {str(e)}")
            continue
    
    raise FileNotFoundError(f"No valid checkpoint found for model index {model_idx}")

def collect_predictions_and_labels(batch_seq, batch_token, out_1, out_2, tokenizer_m2, target_tag):
    """
    Collect predictions and true labels from both m1 and m2 outputs.
    """
    predictions = []
    true_labels = []

    # Token-level predictions (m2)
    predictions_2 = torch.argmax(out_2.logits, dim=2)
    zero_indices = (torch.sigmoid(out_1.logits)[:, 1] < 0.5).nonzero(as_tuple=True)[0]
    
    # Apply sequence-level predictions to token predictions
    for idx in zero_indices:
        predictions_2[idx, :] = 0

    # Process each sequence in the batch
    for i in range(predictions_2.shape[0]):
        # Get token predictions for current sequence
        sequence_preds = predictions_2[i].cpu().numpy()
        
        # Get true labels for current sequence
        if 'labels' in batch_token:
            sequence_labels = batch_token['labels'][i].cpu().numpy()
            # Ensure predictions and labels have same length
            min_len = min(len(sequence_preds), len(sequence_labels))
            predictions.extend(sequence_preds[:min_len])
            true_labels.extend(sequence_labels[:min_len])
        
        # Format spans for string output (if needed)
        if any(sequence_preds == 1):
            token_list = batch_token['tokens'][i]
            text = batch_seq['text'][i]
            article_id = batch_seq['article_id'][i]
            sent_start = batch_seq['sent_start'][i]
            
            try:
                # Process spans here if needed
                # This part remains unchanged
                pass
            except Exception as e:
                logging.warning(f"Error processing spans: {str(e)}")

    return predictions, true_labels

def process_language_chunk(chunk, models, tokenizers, device, target_tag):
    """Process a chunk of data for a specific language."""
    all_predictions = []
    all_true_labels = []
    
    # Prepare datasets
    df_seq = pd.DataFrame(chunk)
    dataset_seq = Dataset.from_pandas(df_seq)
    
    binary_dataset_seq = dataset_seq.map(
        lambda x: tokenize_sequence_classification(x, tokenizers['m1']),
        batched=True,
        batch_size=None,
    )
    
    df_list_binary = span_to_words_annotation(
        dict_of_lists(chunk),
        target_tag=target_tag,
        mappings=regex_tokenizer_mappings,
        labels_model=models['labels']
    )
    df_binary = pd.DataFrame(df_list_binary)
    binary_dataset_token = Dataset.from_pandas(df_binary[TOKEN_COLUMNS])
    
    binary_dataset_token = binary_dataset_token.map(
        lambda x: tokenize_token_classification(x, tokenizers['m2']),
        batched=True,
        batch_size=None,
    )
    
    # Create data loaders
    val_loader_seq = MemoryEfficientDataLoader(
        binary_dataset_seq,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=SequenceCollator(tokenizers['m1'], True, EXCLUSION_COLUMNS)
    )
    
    val_loader_token = MemoryEfficientDataLoader(
        binary_dataset_token,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=TokenCollator(tokenizers['m2'], True, EXCLUSION_COLUMNS)
    )
    
    # Process batches
    for batch_seq, batch_token in tqdm(zip(val_loader_seq, val_loader_token), 
                                     total=len(val_loader_seq),
                                     desc='Processing chunk'):
        try:
            with torch.no_grad():
                # Get M1 predictions
                inputs_seq = {k: v.to(device) for k, v in batch_seq.items() 
                            if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                out_1 = models['m1'](**inputs_seq)

                # Get M2 predictions
                inputs_token = {k: v.to(device) for k, v in batch_token.items() 
                              if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                out_2 = models['m2'](**inputs_token)
                
                # Collect predictions and labels
                batch_preds, batch_labels = collect_predictions_and_labels(
                    batch_seq, batch_token, out_1, out_2, tokenizers['m2'], target_tag
                )
                
                all_predictions.extend(batch_preds)
                all_true_labels.extend(batch_labels)

        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}", exc_info=True)
            continue
    
    return all_predictions, all_true_labels

def main():
    """Main function for running the inference pipeline with improved error handling and logging."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        # Load input data
        json_path = '/home/lgiordano/LUCA/checkthat_GITHUB/data/formatted/train_sentences.json'
        logging.info(f"Loading data from {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Input file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf8') as f:
            data = json.load(f)
            
        if not data:
            raise ValueError("Input data is empty")
        logging.info(f"Successfully loaded {len(data)} samples")

        # Split data
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        logging.info(f"Split data into {len(train_data)} training and {len(test_data)} test samples")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Process each language
        langs = set(el['data']['lang'] for el in test_data)
        for lang in langs:
            logging.info(f'Processing language: {lang}')
            
            # Load M1 model
            path_m1 = CONFIG['model_paths']['m1']
            try:
                tokenizer_m1 = AutoTokenizer.from_pretrained(path_m1)
                m1 = AutoModelForSequenceClassification.from_pretrained(path_m1)
                m1.to(device)
                m1.eval()
                logging.info(f"Successfully loaded M1 model from {path_m1}")
            except Exception as e:
                logging.error(f"Error loading M1 model: {str(e)}")
                continue

            # Filter data for current language
            data_lang = [el for el in test_data if el['data']['lang'] == lang]
            logging.info(f"Processing {len(data_lang)} samples for language {lang}")

            # Initialize containers for all predictions
            all_predictions = []
            all_true_labels = []

            # Define target tags
            target_tags = CONFIG['target_tags']

            # Process each target tag
            model_dir_m2 = CONFIG['model_paths']['m2']
            
            for model_idx, target_tag in enumerate(target_tags[:1]):
                logging.info(f'Processing model {model_idx} of {len(target_tags)-1} for {target_tag}')
                
                try:
                    # Initialize label set
                    labels_model = LabelSet(labels=[target_tag])

                    # Load M2 model
                    path_m2 = find_best_m2_checkpoint(model_dir_m2, model_idx)
                    tokenizer_m2 = AutoTokenizer.from_pretrained(path_m2)
                    m2 = AutoModelForTokenClassification.from_pretrained(
                        path_m2,
                        num_labels=len(labels_model.ids_to_label.values()),
                        label2id=labels_model.labels_to_id,
                        id2label=labels_model.ids_to_label
                    )
                    m2.to(device)
                    m2.eval()
                    logging.info(f"Successfully loaded M2 model from {path_m2}")

                    # Prepare datasets
                    df_seq = pd.DataFrame(data_lang)
                    dataset_seq = Dataset.from_pandas(df_seq)
                    
                    binary_dataset_seq = dataset_seq.map(
                        lambda x: tokenize_sequence_classification(x, tokenizer_m1),
                        batched=True,
                        batch_size=None,
                    )

                    df_list_binary = span_to_words_annotation(
                        dict_of_lists(data_lang),
                        target_tag=target_tag,
                        mappings=regex_tokenizer_mappings,
                        labels_model=labels_model
                    )
                    df_binary = pd.DataFrame(df_list_binary)
                    binary_dataset_token = Dataset.from_pandas(df_binary[TOKEN_COLUMNS])
                    
                    binary_dataset_token = binary_dataset_token.map(
                        lambda x: tokenize_token_classification(x, tokenizer_m2),
                        batched=True,
                        batch_size=None,
                    )

                    # Create data loaders
                    val_loader_seq = DataLoader(
                        binary_dataset_seq,
                        batch_size=16,
                        shuffle=False,
                        collate_fn=SequenceCollator(tokenizer_m1, True, EXCLUSION_COLUMNS)
                    )
                    
                    val_loader_token = DataLoader(
                        binary_dataset_token,
                        batch_size=16,
                        shuffle=False,
                        collate_fn=TokenCollator(tokenizer_m2, True, EXCLUSION_COLUMNS)
                    )

                    # Process batches
                    batch_predictions = []
                    batch_true_labels = []
                    
                    progbar_val = tqdm(
                        zip(val_loader_seq, val_loader_token),
                        total=len(val_loader_seq),
                        desc='Processing batches'
                    )
                    
                    for batch_seq, batch_token in progbar_val:
                        try:
                            with torch.no_grad():
                                # Process sequence classification (M1)
                                inputs_seq = {k: v.to(device) for k, v in batch_seq.items() 
                                           if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                                out_1 = m1(**inputs_seq)

                                # Process token classification (M2)
                                inputs_token = {k: v.to(device) for k, v in batch_token.items() 
                                             if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                                out_2 = m2(**inputs_token)

                                # Collect predictions and labels
                                preds, labels = collect_predictions_and_labels(
                                    batch_seq, batch_token, out_1, out_2, tokenizer_m2, target_tag
                                )
                                
                                # Extend batch results
                                batch_predictions.extend(preds)
                                batch_true_labels.extend(labels)

                        except Exception as e:
                            logging.error(f"Error processing batch: {str(e)}", exc_info=True)
                            continue

                        # Clear CUDA cache periodically
                        if torch.cuda.is_available() and len(batch_predictions) % 1000 == 0:
                            torch.cuda.empty_cache()

                    # Extend all results
                    all_predictions.extend(batch_predictions)
                    all_true_labels.extend(batch_true_labels)

                except Exception as e:
                    logging.error(f"Error processing target tag {target_tag}: {str(e)}", exc_info=True)
                    continue

            # Verify prediction and label counts match
            if len(all_predictions) != len(all_true_labels):
                logging.error(
                    f"Mismatch in predictions ({len(all_predictions)}) "
                    f"and labels ({len(all_true_labels)}) for language {lang}"
                )
                continue

            # Calculate and save metrics
            try:
                metrics = classification_report(
                    y_true=all_true_labels,
                    y_pred=all_predictions,
                    zero_division=0,
                    output_dict=True
                )

                # Save results
                os.makedirs(f'/home/lgiordano/LUCA/checkthat_GITHUB/preds/{date_time}_SIGIR_{CONFIG["aug"][0]}_inference_results', exist_ok=True)
                results_path = f'/home/lgiordano/LUCA/checkthat_GITHUB/preds/{date_time}_SIGIR_{CONFIG["aug"][0]}_inference_results/results_{lang}_{date_time}.json'
                with open(results_path, 'w') as f:
                    json.dump({
                        'language': lang,
                        'metrics': metrics,
                        'total_predictions': len(all_predictions),
                        'total_labels': len(all_true_labels),
                        'timestamp': date_time
                    }, f, indent=2)

                logging.info(f"Results saved to {results_path}")
                logging.info(f"Metrics for language {lang}:")
                logging.info(json.dumps(metrics, indent=2))

            except Exception as e:
                logging.error(f"Error calculating metrics for language {lang}: {str(e)}", exc_info=True)

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()