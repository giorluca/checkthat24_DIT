from tqdm.auto import tqdm
import json
import torch

src_lang = "en"
tgt_lang = "ita_Latn"

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B",
#                                               device_map = 'cuda',
#                                             #   torch_dtype=torch.float16,
#                                               )

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")

filepath = "/home/pgajo/checkthat24/annotated_train.json"

with open(filepath, 'r', encoding='utf8') as f:
    data = json.load(f)

translated_dicts = []
count = 0
for line in tqdm(data, total=len(data)):
    if line['lang'] == src_lang:
        inputs = tokenizer(line['text'], return_tensors="pt")
        inputs = {key: inputs[key].to('cuda') for key in inputs.keys()}
        output = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
        target_text = tokenizer.decode(output.squeeze()[2:])
        line['text'] = target_text
        translated_dicts.append(line)
        count += 1
        if count > 3:
            break

with open(filepath.replace('.json', '_translated.json'), 'w', encoding='utf8') as f:
    json.dump(translated_dicts, f, ensure_ascii = False)