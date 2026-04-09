#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

# DATASET="math"
# MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
# MODEL_BASENAME=$(basename $MODEL_PATH)
# RUN_NAME=${MODEL_BASENAME}_${DATASET}
# NUM_ITER=12 # number of policy gradient inner updates iterations

# accelerate launch \
#     --config_file accelerate.yaml \
#     --main_process_port 12346 diffu_grpo_train.py \
#     --config slurm_scripts/train_dream.yaml \
#     --model_path $MODEL_PATH \
#     --num_iterations $NUM_ITER \
#     --dataset $DATASET \
#     --run_name $RUN_NAME \
#     --output_dir checkpoints/$RUN_NAME \

# run TraDo=4B on 2 GPUs (e.g., A5000)
DATASET="math"
MODEL_PATH=Gen-Verse/TraDo-4B-Instruct
MODEL_BASENAME=$(basename $MODEL_PATH)
RUN_NAME=${MODEL_BASENAME}_${DATASET}
NUM_ITER=12 # number of policy gradient inner updates iterations

accelerate launch \
    --config_file slurm_scripts/accelerate_a5000x2.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train_trado.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \