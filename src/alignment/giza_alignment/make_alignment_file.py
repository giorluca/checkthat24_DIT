import json
import sys
sys.path.append('./src')
from utils_checkthat import TASTEset, sub_shift_spans, regex_tokenizer_mappings, ent_formatter, nllb_lang2code
from tqdm.auto import tqdm
import os
import re

# json_path = './data/train_sent_mt/it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn.json'
# json_path = './data/train_sent_mt/es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn.json'
# json_path = './data/train_sent_mt/bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl.json'
# json_path = './data/train_sent_mt/pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn.json'
# json_path = './data/train_sent_mt/ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl.json'
json_path = './data/train_sent_mt/sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

text_tokenized_src = []
text_tokenized_tgt = []

lang_src = nllb_lang2code[re.search(r'3.3B_(.*)-(.*).json', json_path).group(1)]
lang_tgt = nllb_lang2code[re.search(r'3.3B_(.*)-(.*).json', json_path).group(2)]

langs = [lang_src, lang_tgt]
del_list = [f'annotations_{lang_src}', f'lang_{lang_src}', f'lang_{lang_tgt}']

text_field = 'text'
data_format = 'label_studio'
# data_format = 'tasteset'
strategy = 'regex'

data = TASTEset.checkthat_to_label_studio(data, model_name='gold', lang_list=langs, del_list=del_list, text_field=text_field)

data_tokenized = []
for line in tqdm(data, total=len(data)):
    for lang in langs:
        entry = line
        ents = [ent['value'] for ent in line['annotations'][0]['result'] if ent['from_name'] == f'label_{lang}']
        text, ents = sub_shift_spans(line['data'][f'text_{lang}'],
                            ents=ents,
                            mappings=regex_tokenizer_mappings,
                            )
        entry['data'].update({f'text_{lang}': text})
        if ents:
            entry['annotations'] = [{'result': [ent_formatter(e, lang=lang, text_field=text_field) for e in ents]}]
    data_tokenized.append(entry)

new_dir = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

json_path_new = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}.json')
with open(os.path.join(new_dir, os.path.basename(json_path_new)), 'w', encoding='utf8') as f:
    json.dump(data_tokenized, f, ensure_ascii = False)

if data_format == 'label_studio':
    for line in data_tokenized:
        text_tokenized_src.append(line['data'][f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line['data'][f'{text_field}_{lang_tgt}'])
elif data_format == 'tasteset':
    for line in data_tokenized:
        text_tokenized_src.append(line[f'{text_field}_{lang_src}'])
        text_tokenized_tgt.append(line[f'{text_field}_{lang_tgt}'])

output_filename_en = json_path.replace('.json', f'_tok_{strategy}_{lang_src}_giza.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename_en)), 'w', encoding='utf8') as f:
    for line in text_tokenized_src:
        f.write(line + '\n')

output_filename_it = json_path.replace('.json', f'_tok_{strategy}_{lang_tgt}_giza.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename_it)), 'w', encoding='utf8') as f:
    for line in text_tokenized_tgt:
        f.write(line + '\n')

parallel_list = []

for sent_src, sent_tgt in zip(text_tokenized_src, text_tokenized_tgt):
    parallel_list.append(f'{sent_src} ||| {sent_tgt}')

output_filename = json_path.replace('.json', f'_tok_{strategy}_{lang_src}-{lang_tgt}_fast-align.txt')
with open(os.path.join(new_dir, os.path.basename(output_filename)), 'w', encoding='utf8') as f:
    for line in parallel_list:
        f.write(line + '\n')