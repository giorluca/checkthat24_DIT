import os
import pandas as pd
import json
import logging
import torch
import csv
import sys
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from train_sent import span_to_words_annotation, dict_of_lists
from seq_classification.train_seq import tokenize_sequence_classification
from utils_checkthat import regex_tokenizer_mappings
from labelset import LabelSet
sys.path.append('./src')

def find_best_m2_checkpoint(model_dir: str, model_idx: int) -> str:
    """Find the best checkpoint for a given model index."""
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
                                split_path = path_m2_tmp.split("VAL24/")
                                best_model_checkpoint = split_path[0] + "VAL24/weights/" + split_path[1]                                
                                logging.info(f"Found best checkpoint: {best_model_checkpoint}")
                                return best_model_checkpoint
            except (IndexError, ValueError) as e:
                logging.warning(f"Skipping directory {model_dir_name}: {str(e)}")
    
    logging.error(f"No valid checkpoint found for model index {model_idx}")
    raise FileNotFoundError(f"No valid checkpoint found for model index {model_idx}")

class PersuasionTechniqueDetector:
    def __init__(self, 
                 m1_path: str, 
                 m2_dir: str, 
                 target_tags: List[str]):
        """Initialize the detector with pre-trained models."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Load M1 model (sequence classifier)
        self.tokenizer_m1 = AutoTokenizer.from_pretrained(m1_path)
        self.m1 = AutoModelForSequenceClassification.from_pretrained(m1_path)
        
        # Load M2 models (token classifiers)
        self.m2_models = []
        self.m2_tokenizers = []
        for i in range(len(target_tags)):
            m2_path = find_best_m2_checkpoint(m2_dir, i)
            tokenizer = AutoTokenizer.from_pretrained(m2_path)
            model = AutoModelForTokenClassification.from_pretrained(m2_path)
            self.m2_models.append(model)
            self.m2_tokenizers.append(tokenizer)
        
        self.target_tags = target_tags
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m1.to(self.device)
        for m2 in self.m2_models:
            m2.to(self.device)
        
        # Set models to evaluation mode
        self.m1.eval()
        for m2 in self.m2_models:
            m2.eval()

    def detect_persuasion_techniques(self, text: str) -> List[Dict[str, Any]]:
        """Detect persuasion techniques in the given text."""
        # Sequence classification to filter texts
        seq_inputs = self.tokenizer_m1(text, return_tensors='pt', 
                                       truncation=True, 
                                       max_length=512, 
                                       padding=True)
        seq_inputs = {k: v.to(self.device) for k, v in seq_inputs.items()}
        
        with torch.no_grad():
            seq_outputs = self.m1(**seq_inputs)
        
        # Apply threshold to sequence classification
        threshold = 0.5
        seq_probs = torch.sigmoid(seq_outputs.logits)[:, 1]
        
        # If text doesn't pass sequence classification, return empty list
        if seq_probs < threshold:
            return []
        
        # Detect specific persuasion techniques
        detected_techniques = []
        for i, (m2, m2_tokenizer) in enumerate(zip(self.m2_models, self.m2_tokenizers)):
            # Prepare data for token classification
            labels_model = LabelSet(labels=[self.target_tags[i]])
            tokenized_data = [{
                'tokens': m2_tokenizer.tokenize(text),
                'id': '0',
                'ner_tags': [0] * len(m2_tokenizer.tokenize(text))
            }]
            
            # Tokenize for token classification
            token_inputs = m2_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512, 
                padding=True
            )
            token_inputs = {k: v.to(self.device) for k, v in token_inputs.items()}
            
            # Get token-level predictions
            with torch.no_grad():
                token_outputs = m2(**token_inputs)
            
            # Process predictions
            predictions = torch.argmax(token_outputs.logits, dim=2)
            
            # Convert predictions to spans
            technique_spans = self._extract_spans(
                text, 
                predictions[0], 
                token_inputs['input_ids'][0], 
                m2_tokenizer, 
                self.target_tags[i]
            )
            
            if technique_spans:
                detected_techniques.extend(technique_spans)
        
        return detected_techniques

    def _extract_spans(self, 
                       text: str, 
                       predictions: torch.Tensor, 
                       input_ids: torch.Tensor,
                       tokenizer, 
                       technique: str) -> List[Dict[str, Any]]:
        """Extract text spans for detected persuasion techniques."""
        spans = []
        current_span = None
        
        for i, pred in enumerate(predictions):
            # Skip special tokens (typically 0 and max index)
            if i == 0 or i == len(predictions) - 1:
                continue
            
            # Check if the token is predicted as part of the technique
            if pred != 0:
                # Decode current token
                token_text = tokenizer.decode(input_ids[i:i+1].item()).strip()
                
                # Start of a new span
                if current_span is None:
                    start_idx = text.find(token_text)
                    if start_idx != -1:
                        current_span = {
                            'technique': technique,
                            'start': start_idx,
                            'text': token_text
                        }
                else:
                    # Extend existing span
                    current_span['text'] += token_text
            
            # End of a span
            elif current_span is not None:
                # Complete the span
                current_span['end'] = current_span['start'] + len(current_span['text'])
                spans.append(current_span)
                current_span = None
        
        # Handle span if it ends at the last non-special token
        if current_span is not None:
            current_span['end'] = current_span['start'] + len(current_span['text'])
            spans.append(current_span)
        
        return spans

    def export_results(self, text: str, output_dir: str = None):
        """
        Detect and export persuasion techniques in multiple formats.
        
        :param text: Input text to analyze
        :param output_dir: Directory to save output files (optional)
        :return: Dictionary of results
        """
        # Detect techniques
        results = self.detect_persuasion_techniques(text)
        
        # Prepare result dictionary
        formatted_results = {
            'text': text,
            'techniques': results,
            'summary': {
                'total_techniques': len(results),
                'technique_breakdown': {}
            }
        }
        
        # Count techniques
        for result in results:
            technique = result['technique']
            if technique not in formatted_results['summary']['technique_breakdown']:
                formatted_results['summary']['technique_breakdown'][technique] = 1
            else:
                formatted_results['summary']['technique_breakdown'][technique] += 1
        
        # Export results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # JSON export
            json_path = os.path.join(output_dir, 'persuasion_techniques.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, indent=2, ensure_ascii=False)
            
            # CSV export
            csv_path = os.path.join(output_dir, 'persuasion_techniques.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Technique', 'Span', 'Start', 'End'])
                for result in results:
                    writer.writerow([
                        result['technique'], 
                        result['text'], 
                        result['start'], 
                        result['end']
                    ])
            
            # Text report
            report_path = os.path.join(output_dir, 'persuasion_techniques_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Persuasion Techniques Analysis\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Total Techniques Detected: {len(results)}\n\n")
                f.write("Technique Breakdown:\n")
                for technique, count in formatted_results['summary']['technique_breakdown'].items():
                    f.write(f"- {technique}: {count}\n")
                
                f.write("\nDetailed Findings:\n")
                for result in results:
                    f.write(f"\nTechnique: {result['technique']}\n")
                    f.write(f"Span Text: {result['text']}\n")
                    f.write(f"Start: {result['start']}, End: {result['end']}\n")
        
        return formatted_results

def main():
    # Paths to your pre-trained models
    M1_PATH = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M1/RUN_OTTOBRE/aug/aug, lr 2e-5, + ARAIEVAL(news) & SEMEVAL24/2025-01-16-13-01-16_aug/checkpoint-47691'
    M2_DIR = '/home/lgiordano/LUCA/checkthat_GITHUB/models/M2/RUN_OTTOBRE/weights_and_results/2025-01-13-16-36-34_aug_no_cw_ts0_+ARAIEVAL(news)_&_SEMEVAL24/weights'
    
    # List of persuasion techniques
    TARGET_TAGS = [
        "Appeal_to_Authority", "Appeal_to_Popularity", "Appeal_to_Values", 
        "Appeal_to_Fear-Prejudice", "Flag_Waving", "Causal_Oversimplification",
        "False_Dilemma-No_Choice", "Consequential_Oversimplification", "Straw_Man", 
        "Red_Herring", "Whataboutism", "Slogans", "Appeal_to_Time",
        "Conversation_Killer", "Loaded_Language", "Repetition", 
        "Exaggeration-Minimisation", "Obfuscation-Vagueness-Confusion", 
        "Name_Calling-Labeling", "Doubt", "Guilt_by_Association", 
        "Appeal_to_Hypocrisy", "Questioning_the_Reputation"
    ]
    
    # Initialize the detector
    detector = PersuasionTechniqueDetector(
        m1_path=M1_PATH, 
        m2_dir=M2_DIR, 
        target_tags=TARGET_TAGS
    )
    
    # Data
    df = pd.read_csv('/home/lgiordano/LUCA/CG_EN.csv', sep=',', encoding='utf-8')

    # Detect and export persuasion techniques
    for i, text in enumerate(df['text']):
        output_dir = f'/home/lgiordano/LUCA/CG_EN_persuasion_techniques_output/text_{i}'
        results = detector.export_results(text, output_dir)
        
        # Print summary to console
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()