#!/bin/bash

task='slu'
dataset=$1
read_model_path='' #'exp/task_slu/dataset_atis__model_birnn+crf__labeled_0.1/cell_lstm__emb_400__hidden_256_x_1__dp_0.5__lr_0.001__l2_1e-05__mn_5.0__bs_16__me_50__bm_5__nb_1'
labeled=$2
deviceId=0
seed=999

# model paras
model_type=$3 # birnn, birnn+crf, focus
use_bert='--use_bert' # '--use_bert'
cell=lstm # lstm, gru
emb_size=400
hidden_size=256 # 256, 384
num_layers=1

# training paras
lr=0.001
l2=0.1
layerwise_decay=0.01
warmup_ratio=0.1
lr_schedule=constant
dropout=0.5
batchSize=16
test_batchSize=128
init_weight=0.2
max_norm=5
max_epoch=50
eval_after_epoch=40
beam=5
n_best=1

python3 scripts/slu.py --task $task --dataset $dataset --labeled $labeled --deviceId $deviceId --seed $seed \
    --model_type $model_type $use_bert --cell $cell --emb_size $emb_size --hidden_size $hidden_size --num_layers $num_layers \
    --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --test_batchSize $test_batchSize --init_weight $init_weight \
    --layerwise_decay $layerwise_decay --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule \
    --max_norm $max_norm --max_epoch $max_epoch --eval_after_epoch $eval_after_epoch --beam $beam --n_best $n_best \
    #--testing --read_model_path $read_model_path
