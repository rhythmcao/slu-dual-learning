#!/bin/bash
#SBATCH --job-name=DualLearning
#SBATCH --partition=2080ti
#SBATCH --mem=30g
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH --output=log/dual_%A_%a.out
#SBATCH --error=log/dual_%A_%a.err
#SBATCH --array=0-7

dataset=(atis snips)
cur_d=${dataset[SLURM_ARRAY_TASK_ID / 4]}
ratio=(0.05 0.1)
cur_r=${ratio[(SLURM_ARRAY_TASK_ID % 4) / 2]}
slu_model=(focus bert)
cur_m=${slu_model[SLURM_ARRAY_TASK_ID % 2]}

source activate slu
./run/run_dual_learning.sh ${cur_d} ${cur_r} ${cur_m}
