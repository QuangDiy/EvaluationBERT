#!/bin/bash

MODEL="${MODEL:-QuangDuy/modernbert-tiny-checkpoint-55000ba}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-./all_tasks_results}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
SEED="${SEED:-42}"

TASKS=(mnli qnli rte vnrte wnli sst2 vsfc vsmec mrpc qqp stsb cola vtoc)

mkdir -p $BASE_OUTPUT_DIR

for task in "${TASKS[@]}"; do
    TASK_OUTPUT_DIR="$BASE_OUTPUT_DIR/$task"
    
    python ./ViGLUE/run_viglue.py \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --do_eval \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir $TASK_OUTPUT_DIR \
        --overwrite_output_dir
done

