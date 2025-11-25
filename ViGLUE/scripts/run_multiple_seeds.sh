#!/bin/bash

MODEL="${MODEL:-bert-base-multilingual-cased}"
OUTPUT_DIR="${OUTPUT_DIR:-./multi_seed_results}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"

SEEDS=(1 28 42)
TASKS=(mnli qnli rte vnrte wnli sst2 vsfc vsmec mrpc qqp stsb cola vtoc)

for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running Task: $task | Seed: $seed"
        
        python run_viglue.py \
            --task $task \
            --model_name_or_path $MODEL \
            --do_train \
            --do_eval \
            --num_train_epochs $EPOCHS \
            --per_device_train_batch_size $BATCH_SIZE \
            --seed $seed \
            --output_dir "$OUTPUT_DIR/$task/seed_$seed" \
            --overwrite_output_dir
    done
done
