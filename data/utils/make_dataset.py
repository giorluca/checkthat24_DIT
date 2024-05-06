import os
import pandas as pd
import re
import json
from tqdm.auto import tqdm

def fix_entry(entry):
    for ann in entry['annotations']:
        ann_start = ann['start']
        ann_end = ann['end']
        monitor = entry['text'][ann_end]
        sent = entry['text'][ann['start']:ann['end']]
        if sent[-1] in ',. ':
            ann_end -= 1
        while ann_end < len(entry['text']) and entry['text'][ann_end].isalnum():
            ann_end += 1
            monitor = entry['text'][ann_start:ann_end + 1]
        ann['start_shifted'] = ann_start
        ann['end_shifted'] = ann_end
        sent = entry['text'][ann['start_shifted']:ann['end_shifted']]
    return entry

main_folder = "/home/pgajo/checkthat24/checkthat24_DIT/data/raw"

dataset = []
suffix = ''
# Load gold standard annotations
for lang in os.listdir(main_folder):
    lang_path = os.path.join(main_folder, lang)
    if 'train-labels-subtask-3-spans' in os.listdir(lang_path):
        spans_path = os.path.join(lang_path, f'train-labels-subtask-3-spans{suffix}.json')
        with open(spans_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        if os.path.isdir(lang_path):
            # Iterate over files in train-articles-subtask-3 folder
            train_folder = os.path.join(lang_path, "train-articles-subtask-3")
            sorted_articles = sorted(os.listdir(train_folder))
            for article in tqdm(sorted_articles, total=len(sorted_articles)):
                entry = {}
                if article.endswith(".txt"):
                    article_path = os.path.join(train_folder, article)
                    # Read text from file
                    try:
                        with open(article_path, "r", encoding='utf8') as f:
                            text = f.read()
                        # Extract ID from file name
                        article_id = re.search(r'article(\d+).txt', article).group(1)
                        if article_id in label_data.keys():
                            # some article IDs may not have labels so their label files would have no lines and not appear
                            entry['annotations'] = label_data[article_id]
                        entry['article_id'] = article_id
                        entry['lang'] = lang
                        entry['text'] = text#.replace('\n', '\n ')
                    except UnicodeDecodeError:
                        continue
                entry = fix_entry(entry)
                dataset.append(entry)

train_gold_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold'
dataset_out_path = os.path.join(train_gold_path, f'train_gold_mine.json')
with open(dataset_out_path, 'w', encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii = False)