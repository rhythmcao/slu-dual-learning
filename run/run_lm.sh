#!/bin/bash

task=lm
dataset=$1
read_model_path=''
deviceId=0
seed=999

# model params
cell=lstm # lstm, gru
emb_size=400
hidden_size=400
num_layers=1
decoder_tied='--decoder_tied'
if [ $2 = 'surface' ] ; then
    surface_level='--surface_level'
else
    surface_level=''
fi

# training paras
lr=0.001
l2=1e-5
dropout=0.5
batchSize=16
test_batchSize=128
init_weight=0.2
max_norm=5
max_epoch=100
eval_after_epoch=0

python scripts/lm.py --task $task --dataset $dataset --deviceId $deviceId --seed $seed \
    --num_layers $num_layers --hidden_size $hidden_size --emb_size $emb_size --cell $cell $decoder_tied $surface_level \
    --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --test_batchSize $test_batchSize --init_weight $init_weight \
    --max_norm $max_norm --max_epoch $max_epoch --eval_after_epoch $eval_after_epoch \
    # --testing --read_model_path $read_model_path
