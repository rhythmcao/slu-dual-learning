#!/bin/bash

task=dual_pseudo_labeling
dataset=$1
read_model_path=''
labeled=$2
unlabeled=1.0
deviceIds='0 1'
seed=999

# model paths
if [ ${3} = "bert" ] ; then
    read_slu_model_path="exp/task_slu/dataset_${1}__bert-model_focus__labeled_${2}/cell_lstm__emb_400__hidden_256_x_1__dp_0.5__lr_0.001_ld_0.01_wr_0.1_ls_constant__l2_0.1__mn_5.0__bs_16__me_50__bm_5__nb_1"
else
    read_slu_model_path="exp/task_slu/dataset_${1}__model_focus__labeled_${2}/cell_lstm__emb_400__hidden_256_x_1__dp_0.5__lr_0.001__l2_1e-05__mn_5.0__bs_16__me_50__bm_5__nb_1/"
fi
read_nlg_model_path="exp/task_nlg/dataset_${1}__model_sclstm+copy__labeled_${2}/cell_lstm__emb_400__hidden_256_x_1__sw_1.0__dp_0.5__lr_0.001__l2_1e-05__mn_5.0__bs_16__me_50__bm_5__nb_1"

# training paras
warmup_ratio=0.1 # actually irrelevant due to lr_schedule=constant
lr_schedule=constant
batchSize=16
test_batchSize=128
max_epoch=50
eval_after_epoch=0
beam=5
n_best=1

# special paras
discount=1.0
conf_schedule=linear # constant, linear
cycle_choice=slu+nlg

python3 scripts/dual_pseudo_labeling.py --task $task --dataset $dataset --labeled $labeled --deviceIds $deviceIds --seed $seed \
    --unlabeled $unlabeled --read_slu_model_path $read_slu_model_path --read_nlg_model_path $read_nlg_model_path \
    --batchSize $batchSize --test_batchSize $test_batchSize --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule \
    --max_epoch $max_epoch --eval_after_epoch $eval_after_epoch --beam $beam --n_best $n_best \
    --discount $discount --conf_schedule $conf_schedule --cycle_choice $cycle_choice \
    # --testing --read_model_path $read_model_path
