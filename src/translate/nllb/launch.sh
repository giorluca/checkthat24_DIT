#!/bin/bash
#SBATCH -J checkthat
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/pgajo/checkthat24/.slurm/%j_output.log
#SBATCH --error=/home/pgajo/checkthat24/.slurm/%j_error.log

eval "$(conda shell.bash hook)"
conda activate checkthat

declare -a tgt_languages=(
# 'eng_Latn'

# in xl-wa
'ita_Latn'
# 'spa_Latn'
# 'bul_Cyrl'
# 'por_Latn'
# 'rus_Cyrl'
# 'slv_Latn'
# 'arb_Arab'

# not in xl-wa
# 'ell_Grek'
# 'kat_Geor'
# 'pol_Latn'
# 'deu_Latn'
# 'fra_Latn'
)

# language to translate to (i.e. are in xl-wa)
# langs = [
# 'spa_Latn',
# 'arb_Arab',
# 'por_Latn',
# 'slv_Latn',
# 'bul_Cyrl',
# ]

for tgt_language in "${tgt_languages[@]}";
do
    python ./src/translate/nllb/translate_sent_nllb_batched.py \
    --dataset_path ./data/train_gold/train_gold_sentences.json \
    --train_dir ./data/train_sent_mt \
    --model_name facebook/nllb-200-3.3B \
    --src_lang eng_Latn \
    --tgt_lang $tgt_language
done