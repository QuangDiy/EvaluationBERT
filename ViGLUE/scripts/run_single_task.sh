#!/bin/bash

TASK="${TASK:-mnli}"
MODEL="${MODEL:-bert-base-multilingual-cased}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/$TASK}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
EVAL_BATCH="${EVAL_BATCH:-16}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"

python run_viglue.py \
    --task $TASK \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH \
    --per_device_eval_batch_size $EVAL_BATCH \
    --learning_rate $LR \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir

