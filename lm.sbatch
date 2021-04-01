#!/bin/bash
#SBATCH --job-name=LanguageModel
#SBATCH --partition=2080ti
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --output=log/lm_%A_%a.out
#SBATCH --error=log/lm_%A_%a.err
#SBATCH --array=0-3

dataset=(atis snips)
cur_d=${dataset[SLURM_ARRAY_TASK_ID / 2]}
surface=(sentence surface)
cur_s=${surface[SLURM_ARRAY_TASK_ID % 2]}
source activate slu
./run/run_lm.sh ${cur_d} ${cur_s}
