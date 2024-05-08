import json
from tqdm.auto import tqdm
from evaluate import load
from icecream import ic
import os
import logging
import sys
sys.path.append('/home/pgajo/checkthat24/checkthat24_DIT/src')
from utils_checkthat import get_entities_from_sample, get_relations_from_sample
import uuid
import re
import pandas as pd

def word_idx_to_span(idx, wordlist):
    left = len(''.join(wordlist[:idx])) + len(wordlist[:idx])
    right = left + len(wordlist[idx])
    return {'start': left, 'end': right}

def match_id(ent_src, sample, lang_tgt = 'it'):
    ents = get_entities_from_sample(sample, langs=[lang_tgt])
    rels = get_relations_from_sample(sample)
    from_ids = [rel['from_id'] for rel in rels]
    to_ids = [rel['to_id'] for rel in rels]
    if ent_src['id'] not in from_ids and ent_src['id'] not in to_ids:
        return None
    from_id = ''
    for rel in rels:
        if rel['to_id'] == ent_src['id']:
            from_id = rel['from_id']
            break
    if from_id:
        for ent in ents:
            if ent['id'] == from_id:
                return ent['id']

def format_ls_ent(from_name='', id='', to_name='', type='', start=0, end=0, labels=''):
    ent_ls = {
                    'from_name': from_name,
                    'id': id,
                    'to_name': to_name,
                    'type': type,
                    'value': {
                        'start': start,
                        'end': end,
                        'labels': [labels],
                        }
                    }
    return ent_ls

def pharaoh_aggregate_targets(pharaoh_alignments_line):
    '''
    Given a pharaoh-style line of alignments, e.g.,
    '0-0 1-1 1-2 2-3 3-4 4-5 5-6 5-7 6-8 7-9 8-10'
    aggregate target words aligned to the same source word
    '''
    line_alignments_tuples = [tuple([int(a) for a in line_alignment.split('-')]) for line_alignment in pharaoh_alignments_line.split()]
    alignment_src_last = -1
    alignments_aggregated = []
    alignments_tgt = []
    for alignment in line_alignments_tuples + [(-1,-1)]: # add a dummy tuple to handle the last append
        if alignment[0] != alignment_src_last and alignments_tgt:
            # check if the source entity is different from the last
            # check if we have alignments loaded into alignments_tgt
            alignments_aggregated.append({'source': alignment_src_last, 'target': alignments_tgt})
            alignments_tgt = []
            alignments_tgt.append(alignment[1])
        else:
            alignments_tgt.append(alignment[1])
        alignment_src_last = alignment[0]
    return alignments_aggregated

def flatten_list(input_list):
    input_list = [el for sublist in input_list for el in sublist]
    return input_list

def get_minmax(dict_list):
    values = []
    for el in dict_list:
        values.extend(el.values())
    start = min(values)
    end = max(values)
    return start, end

def match_src_words_to_entities(sample, alignment_list, lang_src = 'en', lang_tgt = 'it', data_name = 'ingredients', model_name = '', append_name = 'annotations'):
    '''
    Aggregate source words falling within the same entity

    `append_name' should be set to 'predictions' if making predictions
    to compare against an already annotated dataset,
    while it can be set to 'annotations' if predicting on a raw dataset
    (or you can choose to put predictions as well in that case) 
    '''
    ent_src_list = get_entities_from_sample(sample, langs=[lang_src], sort=True)
    ingredients_list_src = sample['data'][f'{data_name}_{lang_src}'].split()
    ingredients_list_tgt = sample['data'][f'{data_name}_{lang_tgt}'].split()
    alignments_tgt_final = []
    if append_name == 'predictions':
        sample['predictions'] = [{'model_version': model_name, 'result': [],}]
    elif append_name != 'annotations':
        raise Exception("The name of the annotation column on which to append can only be 'annotations' or 'predictions'.")
    for ent_src in ent_src_list:
        if 'id' not in ent_src.keys():
            print("No id found in source entity, going to assume we are aligning an unannotated dataset.")
            ent_src['id'] = str(uuid.uuid4())
            ent_tgt_id = ent_src['id']
        else:
            ent_tgt_id = match_id(ent_src, sample, lang_tgt=lang_tgt)
        if ent_tgt_id is not None: # check if there is an alignment for this entity
            start_src = ent_src['value']['start']
            end_src = ent_src['value']['end']
            label_src = ent_src['value']['labels'][0]
            for alignment in alignment_list:
                word_src = ingredients_list_src[alignment['source']]
                alignment_src_span = word_idx_to_span(alignment['source'], ingredients_list_src)
                if start_src <= alignment_src_span['start'] and alignment_src_span['end'] <= end_src:
                    alignment_tgt_tmp = [word_idx_to_span(al, ingredients_list_tgt) for al in alignment['target']]
                    alignments_tgt_final.extend(alignment_tgt_tmp)
                    alignment_tgt_tmp = []

            if alignments_tgt_final:
                start_tgt, end_tgt = get_minmax(alignments_tgt_final)
            else:
                start_tgt = start_src
                end_tgt = end_src

            ent_it_new = format_ls_ent(from_name=f'label_{lang_tgt}', id=ent_tgt_id, to_name=f'{data_name}_{lang_tgt}_ref', type='labels', labels=label_src,
                                    start=start_tgt,
                                    end=end_tgt,
                                    )
            sample[append_name][0]['result'].append(ent_it_new)
            alignments_tgt_final = []
    return sample

def evaluate_predictions(dataset, results_path, eval_metric='squad_v2', lang_tgt = 'it', field_src = 'annotations', field_tgt = 'predictions', model_name = ''):
    metric = load(eval_metric)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    preds = []
    trues = []
    type_labels = []

    # Iterate over each sample in the aligned dataset
    for sample in dataset:
        annotations = get_entities_from_sample(sample, langs=[lang_tgt], field=field_src)
        predictions = get_entities_from_sample(sample, langs=[lang_tgt], field=field_tgt)
        # Iterate over each annotation and prediction
        for annotation in annotations:
            for prediction in predictions:
                # Check if the ids match and that the prediction has a non-empty id
                if prediction['id'] == annotation['id'] and prediction['id']:
                    # Prepare the true data format
                    dict_true = {
                        'answers': {
                            'answer_start': [annotation['value']['start']],
                            'text': [sample['data'][f'ingredients_{lang_tgt}'][annotation['value']['start']:annotation['value']['end']]]
                        },
                        'id': annotation['id']
                    }
                    trues.append(dict_true)

                    # Prepare the predicted data format
                    dict_pred = {
                        'prediction_text': sample['data'][f'ingredients_{lang_tgt}'][prediction['value']['start']:prediction['value']['end']],
                        'id': annotation['id'],
                        'no_answer_probability': 0,
                    }
                    preds.append(dict_pred)
                    type_labels.append(annotation['value']['labels'][0])
    types = set(type_labels)
    type_metrics_dict = {type: {} for type in types}
    for type in types:
        preds_type = []
        trues_type = []
        for i in range(len(preds)):
            if type_labels[i] == type:
                preds_type.append(preds[i])
                trues_type.append(trues[i])
        type_metrics_dict[type].update(
            {'f1':metric.compute(predictions=preds_type,
                                            references=trues_type)['f1'],
            'exact':metric.compute(predictions=preds_type,
                                            references=trues_type)['exact'],
            'support': len(preds_type),
            }
        )
    


    results = metric.compute(predictions=preds, references=trues)
    pd.DataFrame(type_metrics_dict).to_csv(f"type_metrics_dict_{model_name}_exact_avg={round(results['exact'], 2)}.csv")
    return results

def main():
    # data_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-it.json'
    # data_path = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es/SW-TASTE_DEEPL_unaligned_ls_tok_regex_en-es.json'
    # data_path = '/home/pgajo/food/data/GZ/GZ-GOLD/GZ-GOLD_301_tok_regex.json'
    # data_path = '/home/pgajo/food/data/mycolombianrecipes/MCR-GOLD_291_tok_regex.json'
    data_path = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt_fixed/it/train_gold_sentences_fixed_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_src-tgt/train_gold_sentences_fixed_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_src-tgt.json'

    with open(data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    # align_folder = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it/giza'
    # align_folder = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-it/fast-align'

    # align_folder = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-es/giza'
    # align_folder = '/home/pgajo/food/data/TASTEset/data/SW-TASTE/en-es/fast-align'

    # align_folder = '/home/pgajo/food/data/GZ/GZ-GOLD/giza'
    # align_folder = '/home/pgajo/food/data/GZ/GZ-GOLD/fast-align'
    
    # align_folder = '/home/pgajo/food/data/mycolombianrecipes/giza'
    # align_folder = '/home/pgajo/food/data/mycolombianrecipes/fast-align'
    
    align_folder = '/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt_fixed/it/giza_alignments'

    append_name = 'annotations'
    # append_name = 'predictions'
    
    data_name = 'text'

    lang_src = 'src'
    lang_tgt = 'tgt'
    lang_id = '-'.join([lang_src, lang_tgt])

    for root, dir, files in os.walk(align_folder):
        files = sorted(files)
        for file in files:
            filename_alignments = os.path.join(root, file)
            model_name = align_folder.split('/')[-1].split('.')[0]
            with open(filename_alignments, 'r', encoding='utf8') as f:
                alignments = f.readlines()

            dataset_aligned = []
            for sample, alignment in tqdm(zip(data, alignments), total=len(data)):
                aggregated_alignments = pharaoh_aggregate_targets(alignment)
                matched_sample = match_src_words_to_entities(sample,
                                                            aggregated_alignments,
                                                            model_name=model_name,
                                                            append_name=append_name,
                                                            data_name=data_name,
                                                            lang_src=lang_src,
                                                            lang_tgt=lang_tgt,
                                                            )
                dataset_aligned.append(matched_sample)
            preds_dir = os.path.join(align_folder, 'preds')
            if not os.path.exists(preds_dir):
                os.makedirs(preds_dir)
            preds_name = os.path.basename(data_path).replace('.json', f'_preds_{model_name}.json')
            dataset_output_path = os.path.join(preds_dir, preds_name)
            with open(dataset_output_path, 'w', encoding='utf8') as f:
                json.dump(dataset_aligned, f, ensure_ascii = False)
            
            print('dataset_output_path:', dataset_output_path)
            
            if append_name == 'predictions': # if we are making 'predictions' for a pre-annotated dataset, then calculate metrics

                results_path = f"./results/alignment/{lang_id}/test/{data_path.split('/')[-1].split('.')[0]}"

                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                results = evaluate_predictions(dataset=dataset_aligned, results_path=results_path, lang_tgt=lang_tgt, field_tgt=append_name, model_name=model_name)
                print(results)
                results_filename = f"{model_name}.json"
                results_output_path = os.path.join(results_path, results_filename)
                with open(results_output_path, 'w', encoding='utf8') as f:
                    json.dump(results, f, ensure_ascii = False)
                
                print('results_output_path:', results_output_path)

if __name__ == "__main__":
    main()