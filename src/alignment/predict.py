from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('/home/pgajo/food/src')
from utils_food import token_span_to_char_indexes, TASTEset
import torch
torch.set_printoptions(linewidth=10000)
import json
from tqdm.auto import tqdm
import re
from icecream import ic

def word_idx_to_span(idx, wordlist):
    left = len(''.join(wordlist[:idx])) + len(wordlist[:idx])
    right = left + len(wordlist[idx])
    return {'start': left, 'end': right}

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-bg/mdeberta_xlwa_en-bg/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-bg_ME3_2024-05-04-11-58-52'
    # model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-es/mdeberta_xlwa_en-es/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-es_ME3_2024-05-04-12-01-43'
    model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-it/mdeberta_xlwa_en-it/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-05-04-12-05-00'
    # model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-pt/mdeberta_xlwa_en-pt/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-pt_ME3_2024-05-04-12-07-45'
    # model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-ru/mdeberta_xlwa_en-ru/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-ru_ME3_2024-05-04-12-09-20'
    # model_name = '/home/pgajo/checkthat24/checkthat24_DIT/models/alignment/en-sl/mdeberta_xlwa_en-sl/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-sl_ME3_2024-05-04-12-12-14'

    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    # json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl.json'
    # json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn.json'
    json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn.json'
    # json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn.json'
    # json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl.json'
    # json_path_unaligned = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn.json'

    with open(json_path_unaligned, 'r', encoding='utf8') as f:
        data = json.load(f)

    lang_src = re.search(r'xlwa_([a-z]{2})-([a-z]{2})_', model_name).group(1)
    lang_tgt = re.search(r'xlwa_([a-z]{2})-([a-z]{2})_', model_name).group(2)

    aligned_dataset = []

    progbar = tqdm(data, total=len(data))
    num_ents = 0
    num_errors = 0
    progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors}')
    for sample in progbar:
        new_sample = sample
        new_annotation_list = []
        for entity in sample['annotations']:
            num_ents += 1
            entity_words = sample['text_src'][entity['start']:entity['end']].split()
            word_span_list = []
            start_list = []
            end_list = []
            for i, word in enumerate(entity_words):
                boundaries = word_idx_to_span(i, entity_words)
                word_span_list.append({'start': boundaries['start'], 'end': boundaries['end']})
            
            for word_span in word_span_list:
                query = sample['text_src'][:word_span['start']] + '• ' + sample['text_src'][word_span['start']:word_span['end']] + ' •' + sample['text_src'][word_span['end']:]
                context = sample['text_tgt']
                input = tokenizer(query, context, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs = model(**input)

                    # set to -10000 any logits in the query (left side of the inputs) so that the model cannot predict those tokens
                    for i in range(len(outputs['start_logits'])):
                        outputs['start_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['start_logits'][i], input['token_type_ids'][i]-10000)
                        outputs['end_logits'][i] = torch.where(input['token_type_ids'][i]!=0, outputs['end_logits'][i], input['token_type_ids'][i]-10000)
                    
                    start_index_token = torch.argmax(outputs['start_logits'], dim=1)

                    # Masking the end_logits to consider only tokens to the right of start_index_token
                    for i in range(len(outputs['end_logits'])):
                        if start_index_token[i] < outputs['end_logits'][i].shape[0]:
                            outputs['end_logits'][i][:start_index_token[i] + 1] = -float('inf')

                    end_index_token = torch.argmax(outputs['end_logits'], dim=1)

                if start_index_token < end_index_token:
                    start, end = token_span_to_char_indexes(input, start_index_token, end_index_token, sample, tokenizer, field_name='text', lang='tgt')
                    start_list.append(start)
                    end_list.append(end)
                elif start_index_token < len(tokenizer(query, return_tensors = 'pt')['input_ids'].squeeze()):
                    print('wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf? wtf?')
                    pass
                else:
                    # print('### START TOKEN !< END TOKEN ###')
                    num_errors += 1
                    pass
            new_annotation = {'start': min(start_list), 'end': max(end_list), 'tag': entity['tag']}
            new_annotation_list.append(new_annotation)
            progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors} - Err%: {round((num_errors/num_ents)*100, 2)}')
            ic(new_sample['text_src'][entity['start']:entity['end']])
            ic(new_sample['text_tgt'][new_annotation['start']:new_annotation['end']])
            print('##########################################')
        new_sample.update({'annotations': new_annotation_list})
        aligned_dataset.append(new_sample)

    filename = f"{json_path_unaligned.replace('.json', '')}_{model_name.split('/')[-1]}.json".replace('unaligned', 'aligned')

    with open(filename, 'w', encoding='utf8') as f:
        json.dump(aligned_dataset, f, ensure_ascii=False)

    print(filename)

    ls_data = TASTEset.tasteset_to_label_studio(data, model_name)

    ls_filename = filename.replace('.json', '_ls.json')
    with open(ls_filename, 'w', encoding='utf8') as f:
        json.dump(ls_data, f, ensure_ascii=False)

if __name__ == '__main__':
    main()