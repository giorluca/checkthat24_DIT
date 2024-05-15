import json
import os
import pandas as pd

path = './models/M2/2024-05-14-06-51-04_aug_ts0.9'

results = []
for subdir in os.listdir(path):
    res = json.load(open(os.path.join(path, subdir, 'results.json')))['results'][-1]
    metrics = list(res.items())[2:6]
    results.append(metrics)

print(results)

# Define the column names
columns = ['persuasion_technique', 'precision', 'recall', 'f1', 'n_samples']

# Initialize an empty list to store the transformed data
transformed_data = []

# Iterate through the original data and transform it
for sublist in results:
    technique_data = {}
    for item in sublist:
        key, value = item
        if '_precision' in key:
            technique = key.split('_precision')[0]
            metric = 'precision'
        elif '_recall' in key:
            technique = key.split('_recall')[0]
            metric = 'recall'
        elif '_f1-score' in key:
            technique = key.split('_f1-score')[0]
            metric = 'f1-score'
        elif '_support' in key:
            technique = key.split('_support')[0]
            metric = 'support'
        if technique not in technique_data:
            technique_data[technique] = {}
        technique_data[technique][metric] = value
    for technique, metrics in technique_data.items():
        transformed_data.append([technique, metrics.get('precision', None), metrics.get('recall', None), metrics.get('f1-score', None), int(metrics.get('support', None))])

# Create a DataFrame
df = pd.DataFrame(transformed_data, columns=columns)

print(df)

print('mean avg f1:', df['f1'].mean())

cols = []
for row in df.iterrows():
    # if int(row[3]) == 0:
    #     cols.append(row[0])
    if row[1]['f1'] == 0:
        cols.append(row[1]['persuasion_technique'])

with open(f"./misc/{path.split('/')[-1]}_bad_models.txt", 'w', encoding='utf8') as f:
    for col in cols:
        f.write(col.strip() + '\n')

df.to_csv(f"./misc/{path.split('/')[-1]}_results.tsv", sep='\t')