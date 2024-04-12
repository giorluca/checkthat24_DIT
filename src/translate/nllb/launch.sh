#!/bin/bash
#SBATCH -J checkthat
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/home/pgajo/checkthat24/.slurm/%j_output.log
#SBATCH --error=/home/pgajo/checkthat24/.slurm/%j_error.log

python /home/pgajo/checkthat24/checkthat24_DIT/src/translate/nllb/translate_nllb.py \
    --model_name facebook/nllb-200-3.3B \
    --src_lang eng_Latn \
    --tgt_lang ita_Latn \