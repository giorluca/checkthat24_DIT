import json
import sys
sys.path.append('./src')
from utils_checkthat import text_to_sentence_sample
from token_classification.train_sent import span_to_words_annotation, dict_of_lists, tokenize_and_align_labels, compute_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

def main():

    json_path = './data/formatted/dev_sentences.json'

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    pass

if __name__ == "__main__":
    main()