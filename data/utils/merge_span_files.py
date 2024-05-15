import os
import re
import json

main_dir = './data/raw'
suffix = ''
split = 'test'
for root, dirs, files in os.walk(main_dir):
    if root.split('/')[-1] == f'{split}-labels-subtask-3-spans':
        filename_merged = f'{root}{suffix}.json'
        with open(filename_merged, 'w', encoding='utf8') as merged_file:
            data = {}
            sorted_files = sorted(os.listdir(root))
            for file in sorted_files:
                if file[0] != '.':
                    m = re.search(r'article(\d+)-labels(-subtask-3)?', file)
                    article_id = m.group(1)
                    filename = os.path.join(root, file)
                    new_content = open(filename, 'r', encoding='utf8').readlines()
                    data[article_id] = []
                    if new_content:
                        for line in new_content:
                            split_line = line.strip().split('\t')
                            if suffix == 'binary':
                                data[article_id].append({'tag': 'persuasion', 'start': int(split_line[2]), 'end': int(split_line[3])})
                            else:
                                data[article_id].append({'tag': split_line[1], 'start': int(split_line[2]), 'end': int(split_line[3])})
            json.dump(data, merged_file, ensure_ascii = False)
