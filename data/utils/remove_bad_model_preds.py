preds_path_list = [
'./preds/2024-05-14-06-51-04_aug_ts0.9/all/all_preds_test_sentences_ar.txt',
'./preds/2024-05-14-06-51-04_aug_ts0.9/all/all_preds_test_sentences_bg.txt',
'./preds/2024-05-14-06-51-04_aug_ts0.9/all/all_preds_test_sentences_en.txt',
'./preds/2024-05-14-06-51-04_aug_ts0.9/all/all_preds_test_sentences_pt.txt',
'./preds/2024-05-14-06-51-04_aug_ts0.9/all/all_preds_test_sentences_sl.txt',
]

bad_models_path = './misc/2024-05-14-06-51-04_aug_ts0.9_bad_models.txt'

with open(bad_models_path, 'r', encoding='utf8') as f:
    data_bad_models = [line.strip() for line in f.readlines()]

for txt_path in preds_path_list:
    with open(txt_path, 'r', encoding='utf8') as f:
        data = [line.strip() for line in f.readlines()]
    
    data_new = []

    for line in data:
        if line.split('\t')[1] not in data_bad_models:
            data_new.append(line)
    
    with open(txt_path.replace('.txt', '_fixed.txt'), 'w', encoding='utf8') as f:
        for line in data_new:
            f.write(line + '\n')

