from tqdm.auto import tqdm
import json
import torch
import re

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model_name = 'facebook/mbart-large-50-many-to-many-mmt'
model_name_simple = model_name.split('/')[-1]
model = MBartForConditionalGeneration.from_pretrained(model_name,
                                                      device_map = 'cuda'
                                                      )

src_lang = 'en_XX'
# tokenizer.src_lang = src_lang
tgt_lang = 'it_IT'

tokenizer = MBart50TokenizerFast.from_pretrained(model_name,
                                                 src_lang=src_lang,
                                                #  tgt_lang=tgt_lang,
                                                 )

filepath = "/home/pgajo/checkthat24/annotated_train.json"

print(tokenizer.src_lang)
print(tokenizer.tgt_lang)

with open(filepath, 'r', encoding='utf8') as f:
    data = json.load(f)

translated_dicts = []
count = 0
batch_size = 16
for line in tqdm(data, total=len(data)):
    if line['lang'] == src_lang.split('_')[0]:
        text_input_list = re.sub(r'\n\n+', r'\n', line['text']).split('\n')
        text_tgt = ''
        for i in range(0, len(text_input_list), batch_size):
            batch = text_input_list[i:i+batch_size]
            #############################
            inputs = tokenizer(batch,
                               return_tensors="pt",
                               padding = 'longest',
                               truncation = True
                               )
            inputs = {k: inputs[k].to('cuda') for k in inputs.keys()}
            if any([len(t.squeeze()) > 1024 for t in inputs['input_ids'].unsqueeze(0)]):
                raise Exception(f"{len(inputs['input_ids'].squeeze())} > 1024 tokens")
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
            )
            text_decoded = '\n'.join(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            text_tgt += text_decoded + '\n'
        line['text'] = text_tgt
        translated_dicts.append(line)
        count += 1
        if count > 3:
            break

with open(filepath.replace('.json', f'_translated_{model_name_simple}.json'), 'w', encoding='utf8') as f:
    json.dump(translated_dicts, f, ensure_ascii = False)