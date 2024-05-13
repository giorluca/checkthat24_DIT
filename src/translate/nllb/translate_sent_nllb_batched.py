from tqdm.auto import tqdm
import json
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import sys
sys.path.append('./src')
from utils_checkthat import nllb_lang2code

def main():
    parser = argparse.ArgumentParser(description="translate checkthat task 3 dataset")
    parser.add_argument("--dataset_path", help="dataset json path", default='./data/formatted/train_sentences.json')
    parser.add_argument("--model_name", help="model huggingface repo name or model dir path", default="facebook/nllb-200-3.3B")
    parser.add_argument("--train_dir", help="path to translated train data directory")
    parser.add_argument("--src_lang", help="source language for dataset filtering", default="eng_Latn")
    parser.add_argument("--tgt_lang", help="target language", default="ita_Latn")
    parser.add_argument('-nv', '--noverbose', action='store_false')
    args = parser.parse_args()

    model_name_simple = args.model_name.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                device_map = 'cuda',
                                                torch_dtype = torch.float16,
                                                )    
    
    print(f"Translating from {args.src_lang} to {args.tgt_lang}...")

    tokenizer.src_lang = args.src_lang        

    with open(args.dataset_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    translated_dicts = []

    data = [sample for sample in data if sample['data']['lang'] == nllb_lang2code[args.src_lang]]
    # data = data[:200]

    batch_size = 64
    for i in tqdm(range(0, len(data), batch_size)):
        # if line['data']['lang'] == nllb_lang2code[args.src_lang]:
        text_tgt = ''
        texts_src = [line['data']['text'] for line in data[i:i+batch_size]]
        article_ids = [line['data']['article_id'] for line in data[i:i+batch_size]]
        annotations = [line['annotations'] for line in data[i:i+batch_size]]
        labels = [line['data']['label'] for line in data[i:i+batch_size]]
        inputs = tokenizer(texts_src,
                        return_tensors="pt",
                        padding = 'longest',
                        truncation = True
                        )
        inputs = {k: inputs[k].to('cuda') for k in inputs.keys()}
        if any([len(input) > 1024 for input in inputs['input_ids'].unsqueeze(0)]):
            raise Exception(f"input length > 1024 tokens")
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang]
        )
        texts_tgt = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        print(texts_tgt)
        for j, text in enumerate(texts_tgt):
            entry = {
                'data': {
                f'text_{nllb_lang2code[args.src_lang]}': texts_src[j],
                f'text_{nllb_lang2code[args.tgt_lang]}': text,
                # f'lang_{nllb_lang2code[args.src_lang]}': args.src_lang,
                # f'lang_{nllb_lang2code[args.tgt_lang]}': args.tgt_lang,
                'article_id': article_ids[j],
                'line_id': i + j,
                'labels': labels[j],
                },
                'annotations': annotations[j],
            }
            
            translated_dicts.append(dict(sorted(entry.items())))

    output_dir = os.path.join(args.train_dir,
                            #   '-'.join([lang_dict[args.src_lang], lang_dict[args.tgt_lang]])
                            )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.basename(args.dataset_path).replace('.json', f'_translated_{model_name_simple}_{args.src_lang}-{args.tgt_lang}.json')
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf8') as f:
        json.dump(translated_dicts, f, ensure_ascii = False)

    print('output_filename', output_filename)

if __name__ == "__main__":
    main()