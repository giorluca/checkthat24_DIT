#!/bin/bash
#SBATCH -J food
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=./.slurm/%j_output.log
#SBATCH --error=./.slurm/%j_error.log

python ./src/token_classification/train_sent.py