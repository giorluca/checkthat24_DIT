import json

data_list = [
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-slv_Latn_tok_regex_en-sl_mdeberta-v3-base_mdeberta_xlwa_en-sl_ME3_2024-05-04-12-12-14_ls.json',
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-rus_Cyrl_tok_regex_en-ru_mdeberta-v3-base_mdeberta_xlwa_en-ru_ME3_2024-05-04-12-09-20_ls.json',
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-por_Latn_tok_regex_en-pt_mdeberta-v3-base_mdeberta_xlwa_en-pt_ME3_2024-05-04-12-07-45_ls.json',
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-ita_Latn_tok_regex_en-it_mdeberta-v3-base_mdeberta_xlwa_en-it_ME3_2024-05-04-12-05-00_ls.json',
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-spa_Latn_tok_regex_en-es_mdeberta-v3-base_mdeberta_xlwa_en-es_ME3_2024-05-04-12-01-43_ls.json',
'/home/pgajo/checkthat24/checkthat24_DIT/data/train_sent_mt/bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg/train_gold_sentences_translated_nllb-200-3.3B_eng_Latn-bul_Cyrl_tok_regex_en-bg_mdeberta-v3-base_mdeberta_xlwa_en-bg_ME3_2024-05-04-11-58-52_ls.json',
]

data_merged = []
langs = [
    'en',
    'it',
    'es',
    'bg',
    'ru',
    'sl',
    'pt',
]

for json_path in data_list:
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    

    