data_path_list = [
'./data/train_sent_mt/sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl_mdeberta-v3-base_mdeberta_xlwa_en-sl_ME3_2024-05-04-12-12-14_ls.json',
'./data/train_sent_mt/ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru_mdeberta-v3-base_mdeberta_xlwa_en-ru_ME3_2024-05-04-12-09-20_ls.json',
'./data/train_sent_mt/pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt_mdeberta-v3-base_mdeberta_xlwa_en-pt_ME3_2024-05-04-12-07-45_ls.json',
'./data/train_sent_mt/it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it_mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-05-04-12-05-00_ls.json',
'./data/train_sent_mt/es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es_mdeberta-v3-base_mdeberta_xlwa_en-es_ME3_2024-05-04-12-01-43_ls.json',
'./data/train_sent_mt/bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg_mdeberta-v3-base_mdeberta_xlwa_en-bg_ME3_2024-05-04-11-58-52_ls.json',
]

import json

for path in data_path_list:
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
        for sample in data:
            for i in range(len(sample['annotations'][0]['result'])):
                if isinstance(sample['annotations'][0]['result'][i], list):
                    if sample['annotations'][0]['result'][i]:
                        sample['annotations'][0]['result'][i] = sample['annotations'][0]['result'][i][0]
                    else:
                        sample['annotations'][0]['result'].remove(sample['annotations'][0]['result'][i])

    json_path = path#.replace('.json', '_fixed.json')

    with open(json_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii = False)