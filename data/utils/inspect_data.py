import json

json_path = './data/train_gold/train_gold_fixed.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)
with open('output.log', 'w', encoding='utf8') as f:
    for sample in data:
        text = sample['text']
        for ann in sample['annotations']:
            start = ann['start']
            # start_shifted = ann['start_shifted']
            end = ann['end']
            # end_shifted = ann['end_shifted']
            # if text[start:end] != text[start_shifted:end_shifted]:
            print('original:', sample['article_id'], sample['lang'], start, end, [text[start:end]], file=f)
            # print('shifted:', sample['article_id'], sample['lang'], start_shifted, end_shifted, [text[start_shifted:end_shifted]], file=f)
            print('-------------------------', file=f)
    f.close()