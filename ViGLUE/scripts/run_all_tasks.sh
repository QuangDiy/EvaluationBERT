#!/bin/bash

MODEL="${MODEL:-QuangDuy/modernbert-tiny-checkpoint-55000ba}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
SEED="${SEED:-42}"

MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +"%Y%m%d_$SEED")
RESULTS_DIR="${RESULTS_DIR:-./results/all_tasks_${MODEL_NAME}_${TIMESTAMP}}"

echo "=========================================="
echo "ViGLUE All Tasks Training"
echo "Model: $MODEL"
echo "Output: $RESULTS_DIR"
echo "=========================================="

mkdir -p "$RESULTS_DIR"

# Task categorization
GLUE_TASKS=(mnli qnli rte wnli sst2 qqp stsb cola mrpc)
TASKS_WITH_TEST=(vsfc vsmec)
TASKS_NO_TEST=(vnrte vtoc)

# Train and test tasks without test set (use validation as test)
for task in "${TASKS_NO_TEST[@]}"; do
    echo "Training $task (validation as test)"
    python ./ViGLUE/run_viglue.py \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --use_validation_as_test \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir
done

# Train and evaluate tasks with test set
for task in "${TASKS_WITH_TEST[@]}"; do
    echo "Training $task (with test set)"
    python ./ViGLUE/run_viglue.py \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --do_eval \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir
done

# Train and evaluate GLUE tasks
for task in "${GLUE_TASKS[@]}"; do
    echo "Training $task (GLUE task)"
    python ./ViGLUE/run_viglue.py \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --do_eval \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir
done
