from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sys
sys.path.append('./src')
from utils_checkthat import token_span_to_char_indexes, TASTEset, get_entities_from_sample
import torch
torch.set_printoptions(linewidth=10000)
import json
from tqdm.auto import tqdm
import re

def word_idx_to_span(word, text = '', text_tokenized = []):
    word_idx = text_tokenized.index(word)
    len_tok = len(' '.join(text_tokenized[:word_idx]))
    text_split = text.split()[:word_idx] 
    len_split = len(' '.join(text_split))
    diff =  len_tok - len_split
    left = len(''.join(text_tokenized[:word_idx])) + len(text_tokenized[:word_idx]) - diff
    right = left + len(word)
    monitor = text[left:right]
    return {'start': left, 'end': right}

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model_name_list = [
    # './models/alignment/en-bg/mdeberta_xlwa_en-bg/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-bg_ME3_2024-05-04-11-58-52',
    # './models/alignment/en-es/mdeberta_xlwa_en-es/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-es_ME3_2024-05-04-12-01-43',
    './models/alignment/en-it/mdeberta_xlwa_en-it/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-05-04-12-05-00',
    # './models/alignment/en-pt/mdeberta_xlwa_en-pt/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-pt_ME3_2024-05-04-12-07-45',
    # './models/alignment/en-ru/mdeberta_xlwa_en-ru/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-ru_ME3_2024-05-04-12-09-20',
    # './models/alignment/en-sl/mdeberta_xlwa_en-sl/mdeberta-v3-base/mdeberta-v3-base_mdeberta_xlwa_en-sl_ME3_2024-05-04-12-12-14',
    ]

    data_path_list = [
    # './data/train_sent_mt/es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es.json',
    './data/train_sent_mt/it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it.json',
    ]

    for model_name, data_path in zip(model_name_list, data_path_list):
        with open(data_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        lang_src = re.search(r'xlwa_([a-z]{2})-([a-z]{2})_', model_name).group(1)
        lang_tgt = re.search(r'xlwa_([a-z]{2})-([a-z]{2})_', model_name).group(2)
        # lang_src = 'src'
        # lang_tgt = 'tgt'
        # langs = [lang_src, lang_tgt]

        aligned_dataset = []

        progbar = tqdm(data, total=len(data))
        num_ents = 0
        num_errors = 0
        progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors}')
        for sample in progbar:
            new_sample = sample
            annotation_list_tgt = []
            entity_list = get_entities_from_sample(sample, langs=[lang_src], sort=True)
            for entity in entity_list:
                num_ents += 1
                # get words comprising the entity
                entity_words = sample['data'][f'text_{lang_src}'][entity['value']['start']:entity['value']['end']].split()
                word_span_list = []
                start_list = []
                end_list = []
                for i, word in enumerate(entity_words):
                    left = sample['data'][f'text_{lang_src}'][entity['value']['start']:entity['value']['end']].find(word) + entity['value']['start']
                    right = left + len(word)
                    # monitor = sample['data'][f'text_{lang_src}'][left:right]
                    word_span_list.append({'start': left, 'end': right})
                
                for word_span in word_span_list:
                    text_left = sample['data'][f'text_{lang_src}'][:word_span['start']]
                    text_right = sample['data'][f'text_{lang_src}'][word_span['end']:]
                    text_entity = sample['data'][f'text_{lang_src}'][word_span['start']:word_span['end']]
                    query = text_left + '• ' + text_entity + ' •' + text_right
                    context = sample['data'][f'text_{lang_tgt}']
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
                        start, end = token_span_to_char_indexes(input, start_index_token, end_index_token, sample['data'], tokenizer, field_name='text', lang=lang_tgt)
                        start_list.append(start)
                        end_list.append(end)
                    elif start_index_token < len(tokenizer(query, return_tensors = 'pt')['input_ids'].squeeze()):
                        raise Exception('start_index_token predicted within sequence A (i.e. the query)')
                    else:
                        # print('### START TOKEN !< END TOKEN ###')
                        num_errors += 1
                if start_list and end_list:
                    new_annotation_tgt = {'start': min(start_list), 'end': max(end_list), 'tag': entity['value']['labels'][0]}
                    annotation_list_tgt.append(new_annotation_tgt)
                progbar.set_description(f'Entities: {num_ents} - Errors: {num_errors} - Err%: {round((num_errors/num_ents)*100, 2)}')
                # print(new_sample['data'][f'text_{lang_src}'][entity['value']['start']:entity['value']['end']])
                # print(new_sample['data'][f'text_{lang_tgt}'][new_annotation_tgt['start']:new_annotation_tgt['end']])
                # print('##########################################')
            new_sample[f'annotations_{lang_tgt}'] = [{'result': annotation_list_tgt}]
            aligned_dataset.append(new_sample)

        filename = f"{data_path.replace('.json', '')}_{model_name.split('/')[-1]}.json".replace('unaligned', 'aligned')

        with open(filename, 'w', encoding='utf8') as f:
            json.dump(aligned_dataset, f, ensure_ascii=False)

        print(filename)

        ls_data = TASTEset.tasteset_to_label_studio(data, model_name)

        ls_filename = filename.replace('.json', '_ls.json')
        with open(ls_filename, 'w', encoding='utf8') as f:
            json.dump(ls_data, f, ensure_ascii=False)

if __name__ == '__main__':
    main()