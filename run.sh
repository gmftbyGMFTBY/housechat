#!/bin/bash

# ./run.sh train bert 2,3

mode=$1
model=$2
cuda=$3

models=(bert)
for m in ${models[@]}
do
    mkdir -p ckpt/$m
    mkdir -p rest/$m
    mkdir -p bak/$m
done

if [ $mode = 'train' ]; then
    gpu_ids=(${cuda//,/ })
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 29400 main.py \
            --model $model \
            --mode train \
            --batch_size 32 \
            --epoch 5 \
            --seed 50 \
            --max_length 256 \
            --multi_gpu $cuda
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES=$cuda python main.py \
        --model $model \
        --mode test \
        --batch_size 1 \
        --max_length 256 \
        --seed 50 \
        --multi_gpu $cuda
else
    echo '[!] wrong mode find: $mode'
fi