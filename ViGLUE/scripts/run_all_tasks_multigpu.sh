#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (two levels up from scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_GPUS="${NUM_GPUS:-2}"
MODEL="${MODEL:-QuangDuy/modernbert-tiny-checkpoint-55000ba}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
SEED="${SEED:-42}"

MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +"%Y%m%d_$SEED")
RESULTS_DIR="${RESULTS_DIR:-./results/all_tasks_${MODEL_NAME}_${TIMESTAMP}_${NUM_GPUS}gpu}"

echo "=========================================="
echo "ViGLUE All Tasks Training (Multi-GPU)"
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Batch size per GPU: $TRAIN_BATCH"
echo "Effective batch size: $((TRAIN_BATCH * NUM_GPUS))"
echo "Output: $RESULTS_DIR"
echo "=========================================="

mkdir -p "$RESULTS_DIR"
SUBMISSION_DIR="$RESULTS_DIR/submissions"
mkdir -p "$SUBMISSION_DIR"

GLUE_TASKS=(mnli qnli rte wnli sst2 qqp cola mrpc)
TASKS_WITH_TEST=(vsfc vsmec)
TASKS_NO_TEST=(vnrte vtoc)

for task in "${TASKS_NO_TEST[@]}"; do
    echo "Training $task (validation as test) on $NUM_GPUS GPUs"
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        "$PROJECT_ROOT/ViGLUE/run_viglue.py" \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --use_validation_as_test \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir \
        --no_save_model \
        --no_timestamp_dir
done

# Train and evaluate tasks with test set
for task in "${TASKS_WITH_TEST[@]}"; do
    echo "Training $task (with test set) on $NUM_GPUS GPUs"
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        "$PROJECT_ROOT/ViGLUE/run_viglue.py" \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --do_eval \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir \
        --no_save_model \
        --no_timestamp_dir
done

# Train and generate predictions for GLUE tasks
for task in "${GLUE_TASKS[@]}"; do
    echo "Training $task (GLUE task - will generate submission files) on $NUM_GPUS GPUs"
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        "$PROJECT_ROOT/ViGLUE/run_viglue.py" \
        --task $task \
        --model_name_or_path $MODEL \
        --do_train \
        --do_predict \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BATCH \
        --seed $SEED \
        --output_dir "$RESULTS_DIR/$task" \
        --overwrite_output_dir \
        --no_save_model \
        --no_timestamp_dir
done

echo "Collecting GLUE submission files..."
for task in "${GLUE_TASKS[@]}"; do
    task_dir="$RESULTS_DIR/$task"
    if [ -d "$task_dir" ]; then
        find "$task_dir" -name "*.tsv" -exec cp {} "$SUBMISSION_DIR/" \;
    fi
done

echo "Generating mock submission files for STS-B, AX..."
python "$PROJECT_ROOT/ViGLUE/generate_mock_submissions.py" \
    --output-dir "$SUBMISSION_DIR" \
    --seed $SEED

if ls "$SUBMISSION_DIR"/*.tsv 1> /dev/null 2>&1; then
    cd "$SUBMISSION_DIR" && zip -q submission.zip *.tsv && cd - > /dev/null
    echo "Submission zip created: $SUBMISSION_DIR/submission.zip"
fi

echo "=========================================="
echo "All tasks completed!"
echo "Results: $RESULTS_DIR"
echo "=========================================="
