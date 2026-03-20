#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET="gsm8k"
RUN_NAME=${DATASET}_dream_bs12
MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
NUM_ITER=12 # number of policy gradient inner updates iterations
MASK_ID=151669 # Dream mask token id (<|mask|>)

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --mask_id $MASK_ID
