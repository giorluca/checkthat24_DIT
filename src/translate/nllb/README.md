
# NLLB Translation

This repository provides scripts for batched sentence translation using the NLLB (No Language Left Behind) model. The script `translate_sent_nllb_batched.py` allows you to translate sentences in batches.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Requirements

- Python 3.6+
```bash
pip install transformers torch tqdm
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/paolo-gajo/nllb-translation.git
cd nllb-translation
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the batched sentence translation script, run the following command:

```bash
python translate_sent_nllb_batched.py --dataset_path <path_to_dataset> --model_name <model_name_or_path> --train_dir <output_directory> --src_lang <source_language> --tgt_lang <target_language>
```

### Arguments

- `--dataset_path`: Path to the dataset JSON file. Default is `./data/formatted/train_sentences.json`.
- `--model_name`: HuggingFace model name or directory path of a local model. Default is `facebook/nllb-200-3.3B`.
- `--train_dir`: Path to the directory where the translated data will be saved.
- `--src_lang`: Source language code for dataset filtering. Default is `eng_Latn`.
- `--tgt_lang`: Target language code for translation. Default is `ita_Latn`.
- `--noverbose`: Disable verbose output.

### Example Command

```bash
python translate_sent_nllb_batched.py --dataset_path ./data/formatted/train_sentences.json --model_name facebook/nllb-200-3.3B --train_dir ./translated_data --src_lang eng_Latn --tgt_lang ita_Latn
```

## Examples

The following is an example of how to use the script:

```bash
python translate_sent_nllb_batched.py --dataset_path ./data/sample_dataset.json --model_name facebook/nllb-200-1.3B --train_dir ./output --src_lang fra_Latn --tgt_lang spa_Latn
```

This command will translate sentences from French to Spanish using the NLLB model and save the translated dataset in the specified output directory.
