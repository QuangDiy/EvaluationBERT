#!/bin/bash

# MODEL="${MODEL:-QuangDuy/modernbert-tiny-checkpoint-55000ba}"
MODEL="${MODEL:-jhu-clsp/mmBERT-base}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-42}"

MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +"%Y%m%d_$SEED")
RESULTS_DIR="${RESULTS_DIR:-./results/baseline_${MODEL_NAME}_${TIMESTAMP}}"
SUBMISSION_DIR="$RESULTS_DIR/submissions"

echo "=========================================="
echo "ViGLUE Baseline Evaluation"
echo "Model: $MODEL"
echo "Output: $RESULTS_DIR"
echo "=========================================="

mkdir -p "$RESULTS_DIR"
mkdir -p "$SUBMISSION_DIR"

GLUE_TASKS=(mnli qnli rte wnli sst2)
# GLUE_TASKS=(mnli qnli rte wnli sst2 qqp cola mrpc)
TASKS_WITH_TEST=(vsfc vsmec)
TASKS_NO_TEST=(vnrte vtoc)

for task in "${GLUE_TASKS[@]}"; do
    echo "$task submission"
    python ./ViGLUE/generate_glue_submission.py \
        --task $task \
        --model-path $MODEL \
        --output-dir $SUBMISSION_DIR \
        --batch-size $BATCH_SIZE
done

echo "Generating mock submission files for STS-B, AX..."
python ./ViGLUE/generate_mock_submissions.py \
    --output-dir $SUBMISSION_DIR \
    --seed $SEED

for task in "${TASKS_WITH_TEST[@]}"; do
    echo "$task (test)"
    python ./ViGLUE/run_baseline.py \
        --task $task \
        --model_name_or_path $MODEL \
        --batch_size $BATCH_SIZE \
        --eval_split test \
        --output_dir "$RESULTS_DIR/${task}" \
        --seed $SEED
done

for task in "${TASKS_NO_TEST[@]}"; do
    echo "$task (val)"
    python ./ViGLUE/run_baseline.py \
        --task $task \
        --model_name_or_path $MODEL \
        --batch_size $BATCH_SIZE \
        --eval_split validation \
        --output_dir "$RESULTS_DIR/${task}" \
        --seed $SEED
done

if ls "$SUBMISSION_DIR"/*.tsv 1> /dev/null 2>&1; then
    cd "$SUBMISSION_DIR" && zip -q submission.zip *.tsv && cd - > /dev/null
    echo "Submission zip created: $SUBMISSION_DIR/submission.zip"
fi
