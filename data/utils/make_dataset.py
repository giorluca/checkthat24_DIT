import os
import re
import json
from tqdm.auto import tqdm
import unicodedata

def sub_shift_spans(text, ents, mappings = []):
    for mapping in mappings:
        adjustment = 0
        pattern = re.compile(mapping['pattern'])
        for match in re.finditer(pattern, text):
            match_index = match.start() + adjustment
            match_contents = match.group()
            if all(unicodedata.category(char).startswith('P') for char in match_contents):
                subbed_text = mapping['target'].replace('placeholder', match_contents)
            else:
                subbed_text = match_contents
            len_diff = len(subbed_text) - len(match_contents)
            text = text[:match_index] + subbed_text + text[match_index + len(match_contents):]

            if isinstance(ents, list):
                for ent in ents:
                    if ent['start'] <= match_index and ent['end'] > match_index:
                        ent['end'] += len_diff
                    if ent['start'] > match_index:
                        ent['start'] += len_diff
                        ent['end'] += len_diff
            elif isinstance(ents, dict):
                if ents['start'] <= match_index and ents['end'] > match_index:
                    ents['end'] += len_diff
                if ents['start'] > match_index:
                    ents['start'] += len_diff
                    ents['end'] += len_diff

            adjustment += len_diff
    # for ent in ent_list:
    #     ic(text[ent['start']:ent['end']])
    return text, ents

main_folder = "./data/raw"

mappings = [
    {'pattern': r'\r\n', 'target': r'\n'},
]

dataset = []
suffix = ''
split = 'test'
annotated = 1

# Load gold standard annotations
for lang in os.listdir(main_folder):
    lang_path = os.path.join(main_folder, lang)
    if annotated:
        if f'{split}-labels-subtask-3-spans' in os.listdir(lang_path):
            spans_path = os.path.join(lang_path, f'{split}-labels-subtask-3-spans{suffix}.json')
            with open(spans_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
    else:
        label_data = {}
    article_dir_name = f"{split}-articles-subtask-3"
    if article_dir_name in os.listdir(lang_path):
        # Iterate over files in train-articles-subtask-3 folder
        article_dir_path = os.path.join(lang_path, article_dir_name)
        sorted_articles = sorted(os.listdir(article_dir_path))
        for article in tqdm(sorted_articles, total=len(sorted_articles)):
            entry = {}
            if article.endswith(".txt"):
                article_path = os.path.join(article_dir_path, article)
                # Read text from file
                try:
                    with open(article_path, "rb") as f:
                        text = f.read().decode('utf-8') 
                    # Extract ID from file name
                    article_id = re.search(r'article(\d+).txt', article).group(1)
                    if article_id in label_data.keys():
                        # some article IDs may not have labels so their label files would have no lines and not appear
                        entry['annotations'] = sorted(label_data[article_id], key=lambda x: x['start'])
                    else:
                        entry['annotations'] = []
                    entry['article_id'] = article_id
                    entry['lang'] = lang
                    entry['text'] = text
                    entry['text'], entry['annotations'] = sub_shift_spans(text, entry['annotations'], mappings=mappings)
                    with open('output.log', 'w', encoding='utf8') as f:
                        print(entry['text'], file=f)
                        print('', file=f)
                        for ann in entry['annotations']:
                            print(article_id, ann['start'], ann['end'], entry['text'][ann['start']:ann['end']], file = f)
                except UnicodeDecodeError:
                    continue
            dataset.append(entry)

out_dir_path = f'./data/formatted'
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)
dataset_out_path = os.path.join(out_dir_path, f'{split}.json')
with open(dataset_out_path, 'w', encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii = False)