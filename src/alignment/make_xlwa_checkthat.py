import os
import sys
sys.path.append('./src')
from utils_checkthat import XLWADataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets

def main():
    data_path = '/home/pgajo/food/data/XL-WA/data'
    languages = [
    #   'ru',
    #   'nl',
    #   'it',
    #   'pt',
    #   'et',
    #   'es',
    #   'hu',
    #   'da',
    #   'bg',
      'sl',
      ]

    tokenizer_name = 'microsoft/mdeberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    split_mapping = {'train': ['dev', 'test'], 'dev': ['dev'], 'test': ['test']}
    dataset = XLWADataset(
        data_path,
        tokenizer,
        languages = languages,
        split_mapping = split_mapping
        )
    

    tokenizer_dict = {
        'bert-base-multilingual-cased': 'mbert',
        'microsoft/mdeberta-v3-base': 'mdeberta',
    }
    save_name = f"{tokenizer_dict[tokenizer_name]}_{dataset.name}"
    repo_id = f"pgajo/{save_name}"
    print('repo_id:', repo_id)

    datasets_dir_path = f"./data/alignment/train/en-{'-'.join(languages)}/{type(tokenizer).__name__}/{tokenizer_name.split('/')[-1]}"
    if not os.path.exists(datasets_dir_path):
        os.makedirs(datasets_dir_path)
    full_save_path = os.path.join(datasets_dir_path, save_name)
    print(full_save_path)
    dataset.save_to_disk(full_save_path)

if __name__ == '__main__':
    main()