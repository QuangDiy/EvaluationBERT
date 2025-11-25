#!/bin/bash

MODEL="${MODEL:-bert-base-multilingual-cased}"
OUTPUT_DIR="${OUTPUT_DIR:-./baseline_results}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"

python run_baseline.py \
    --all_tasks \
    --model_name_or_path $MODEL \
    --batch_size $BATCH_SIZE \
    --eval_split validation \
    --output_dir $OUTPUT_DIR \
    --seed $SEED

