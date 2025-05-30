#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

dataset=$1
split=${2:-"test"}

model_path="model_zoo/MUPA_2B"

pred_path="outputs/${dataset}_${split}_MUPA_2B"
echo "pred_path: $pred_path"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

trap 'echo "Caught SIGINT, terminating all processes."; kill $(jobs -p) && wait $(jobs -p); exit 1' SIGINT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto_multipath.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_path $model_path \
        --chunk $CHUNKS \
        --index $IDX \
        --task GQA &

done

wait

python videomind/eval/eval_multipath.py --pred_path $pred_path --dataset $dataset
