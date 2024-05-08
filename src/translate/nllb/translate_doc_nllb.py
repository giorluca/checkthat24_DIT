from tqdm.auto import tqdm
import json
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def main():
    parser = argparse.ArgumentParser(description="translate checkthat task 3 dataset")
    parser.add_argument("--dataset_path", help="source language", default="/home/pgajo/checkthat24/data/annotated_train.json")
    parser.add_argument("--model_name", help="model huggingface repo name or model dir path", default="facebook/nllb-200-3.3B")
    parser.add_argument("--train_dir", help="path to translated train data directory", default='./data/train')
    parser.add_argument("--src_lang", help="source language", default="eng_Latn")
    parser.add_argument("--tgt_lang", help="target language", default="ita_Latn")
    args = parser.parse_args()

    lang_dict = {
        "eng_Latn": 'en',
        "ita_Latn": 'it',
        'rus_Cyrl': 'ru',
        'spa_Latn': 'es',
        'arb_Arab': 'ar',
        'por_Latn': 'po',
        'slv_Latn': 'sl',
        'bul_Cyrl': 'bg',
    }

    model_name_simple = args.model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                device_map = 'cuda',
                                                torch_dtype = torch.float16,
                                                #   attn_implementation="flash_attention_2", # not implemented yet for M2M100ForConditionalGeneration
                                                )

    with open(args.dataset_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    data = [line for line in data if line['lang'] == lang_dict[args.src_lang]]

    translated_dicts = []
    count = 0
    batch_size = 4
    for line in tqdm(data, total=len(data)):
        text_input_list = [text for text in re.sub(r'\n\n+', r'\n', line['text']).split('\n') if text] # remove any possible empty lines
        text_tgt = ''
        for i in range(0, len(text_input_list), batch_size):
            batch = text_input_list[i:i+batch_size]
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
                forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang]
            )
            text_decoded = '\n'.join(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
            text_tgt += text_decoded + '\n'
        line['text_src'] = line['text'].strip()
        line['text_tgt'] = text_tgt.strip()
        line['lang_src'] = args.src_lang
        line['lang_tgt'] = args.tgt_lang
        line.pop('lang')
        line.pop('text')
        translated_dicts.append(dict(sorted(line.items())))
        # count += 1
        # if count > 1:
        #     break

    output_dir = os.path.join(args.train_dir, '-'.join([lang_dict[args.src_lang], lang_dict[args.tgt_lang]]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.basename(args.dataset_path).replace('.json', f'_translated_{model_name_simple}.json')
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf8') as f:
        json.dump(translated_dicts, f, ensure_ascii = False)

    print('output_filename', output_filename)

if __name__ == "__main__":
    main()