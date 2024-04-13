#!/bin/bash
#SBATCH -J checkthat
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/pgajo/checkthat24/.slurm/%j_output.log
#SBATCH --error=/home/pgajo/checkthat24/.slurm/%j_error.log

declare -a tgt_languages=(
# 'ita_Latn'
'rus_Cyrl'
'spa_Latn'
'arb_Arab'
'por_Latn'
'slv_Latn'
'bul_Cyrl'
# 'ell_Grek'
# 'kat_Geor'
# 'pol_Latn'
# 'deu_Latn'
# 'fra_Latn'
)

for tgt_language in "${tgt_languages[@]}";
do
    python /home/pgajo/checkthat24/checkthat24_DIT/src/translate/nllb/translate_nllb.py \
    --model_name facebook/nllb-200-3.3B \
    --src_lang eng_Latn \
    --tgt_lang $tgt_language
done