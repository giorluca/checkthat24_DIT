import re
import json
import sys
sys.path.append('./src')
from utils_checkthat import text_to_sentence_sample
import argparse

def main():
    parser = argparse.ArgumentParser(description="make checkthat24 sentence dataset from doc dataset")
    parser.add_argument("--input", "-i", help="json doc dataset path", default='./data/formatted/dev.json')
    args = parser.parse_args()
    data = json.load(open(args.input, 'r', encoding='utf8'))

    sentence_data = []
    for sample in data:
        sentence_data.extend(text_to_sentence_sample(sample, lang=sample['lang']))
    out_path = args.input.replace('.json', '_sentences.json')
    with open(out_path, 'w', encoding='utf-8') as outfile: 
        json.dump(sentence_data, outfile, ensure_ascii = False)

if __name__ == '__main__':
    main()
