import sys
sys.path.append('./src')
from utils_checkthat import TASTEset
import json

json_path = './data/formatted/train_sentences.json'

with open(json_path, 'r', encoding='utf8') as f:
    data = json.load(f)

data_ls = TASTEset.checkthat_to_label_studio(data, lang_list=['de'])

pass