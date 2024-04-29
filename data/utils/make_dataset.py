import os
import pandas as pd
import re
import json

main_folder = "/home/pgajo/checkthat24/checkthat24_DIT/data/raw_mine"

dataset = []
suffix = 'binary'
# Load gold standard annotations
for lang in os.listdir(main_folder):
    lang_path = os.path.join(main_folder, lang)
    if 'train-labels-subtask-3-spans' in os.listdir(lang_path):
        spans_path = os.path.join(lang_path, f'train-labels-subtask-3-spans_{suffix}.json')
        with open(spans_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        if os.path.isdir(lang_path):
            # Iterate over files in train-articles-subtask-3 folder
            train_folder = os.path.join(lang_path, "train-articles-subtask-3")
            for file in os.listdir(train_folder):
                entry = {}
                if file.endswith(".txt"):
                    file_path = os.path.join(train_folder, file)
                    # Extract ID from file name
                    article_id = re.search(r'article(\d+).txt', file).group(1)
                    if article_id in label_data.keys():
                        # some article IDs may not have labels so their label files would have no lines and not appear
                        entry['annotations'] = label_data[article_id]
                    # Read text from file
                    try:
                        with open(file_path, "r", encoding='utf8') as f:
                            text = f.read()
                        entry['article_id'] = article_id
                        entry['lang'] = lang
                        entry['text'] = text
                    except UnicodeDecodeError:
                        pass
                dataset.append(entry)

train_gold_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_gold'
dataset_out_path = os.path.join(train_gold_path, f'train_gold_{suffix}.json')
with open(dataset_out_path, 'w', encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii = False)