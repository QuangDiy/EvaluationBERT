# ViGLUE Evaluation Framework

A comprehensive framework for evaluating BERT models on the Vietnamese General Language Understanding Evaluation (ViGLUE) benchmark using HuggingFace Transformers.

## Features

- **13 ViGLUE Tasks**: Support for all tasks including MNLI, QNLI, RTE, VNRTE, WNLI, SST2, VSFC, VSMEC, MRPC, QQP, STS-B, CoLA, and VToC
- **Multiple Evaluation Modes**: Fine-tuning, baseline (zero-shot), and batch experiments
- **Robust Evaluation**: Multiple random seeds with mean/std aggregation
- **Configurable Hyperparameters**: Batch size, learning rate, epochs, and more
- **Comprehensive Metrics**: Accuracy, F1, Matthews Correlation, Pearson/Spearman correlations
- **Well-Structured**: Modular design with separate config, utils, and trainers modules

## Project Structure

```
ViGLUE/
├── config/
│   ├── __init__.py
│   ├── task_config.py          # Task-specific configurations
│   └── training_config.py      # Training hyperparameters
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading utilities
│   ├── model_utils.py          # Model initialization
│   ├── metrics.py              # Evaluation metrics
│   └── logging_utils.py        # Logging and result saving
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py         # Training logic
│   └── evaluator.py            # Evaluation logic
├── run_viglue.py               # Main fine-tuning script
├── run_baseline.py             # Baseline evaluation script
├── run_experiments.py          # Batch experiments script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone or navigate to the ViGLUE directory**:
   ```bash
   cd d:\workspace\EvaluationBERT\ViGLUE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import transformers, datasets, torch; print('✓ All dependencies installed')"
   ```

## Usage

### 1. Fine-tuning on a Single Task

Fine-tune a BERT model on a specific ViGLUE task:

```bash
python run_viglue.py \
    --task mnli \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/mnli
```

#### Available Tasks

| Task | Type | Metric | Description |
|------|------|--------|-------------|
| `mnli` | NLI | Accuracy | Multi-Genre Natural Language Inference |
| `qnli` | NLI | Accuracy | Question Natural Language Inference |
| `rte` | NLI | Accuracy | Recognizing Textual Entailment |
| `vnrte` | NLI | Accuracy | Vietnamese RTE |
| `wnli` | NLI | Accuracy | Winograd NLI |
| `sst2` | Sentiment | Accuracy | Stanford Sentiment Treebank |
| `vsfc` | Sentiment | Accuracy | Vietnamese Student Feedback |
| `vsmec` | Sentiment | Accuracy | Vietnamese Social Media Emotion |
| `mrpc` | Similarity | Acc/F1 | Microsoft Research Paraphrase Corpus |
| `qqp` | Similarity | Acc/F1 | Quora Question Pairs |
| `stsb` | Similarity | Pearson/Spearman | Semantic Textual Similarity |
| `cola` | Acceptability | MCC | Corpus of Linguistic Acceptability |
| `vtoc` | Classification | Accuracy | Vietnamese Topic Classification |

#### Key Arguments

**Model and Task**:
- `--task`: Task name (required)
- `--model_name_or_path`: Pretrained model name or path (default: `bert-base-multilingual-cased`)

**Execution Flags**:
- `--do_train`: Enable training
- `--do_eval`: Enable evaluation
- `--do_predict`: Generate predictions

**Training Hyperparameters**:
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Training batch size (default: 32)
- `--per_device_eval_batch_size`: Evaluation batch size (default: 64)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--warmup_ratio`: Warmup ratio (default: 0.1)
- `--weight_decay`: Weight decay (default: 0.01)
- `--max_seq_length`: Maximum sequence length (default: 128)

**Other Options**:
- `--seed`: Random seed (default: 42)
- `--fp16`: Use mixed precision training
- `--quick_test`: Quick test mode (1 epoch)
- `--output_dir`: Output directory for results

### 2. Baseline Evaluation (No Fine-tuning)

Evaluate a pretrained model without fine-tuning to establish baseline performance:

```bash
# Single task
python run_baseline.py \
    --task mnli \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir ./baseline_results

# All tasks
python run_baseline.py \
    --all_tasks \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir ./baseline_results
```

#### Arguments

- `--task`: Single task to evaluate
- `--all_tasks`: Evaluate on all ViGLUE tasks
- `--model_name_or_path`: Model to evaluate
- `--batch_size`: Batch size for evaluation (default: 64)
- `--eval_split`: Dataset split to evaluate on (`validation`, `test`, or `all`)
- `--output_dir`: Output directory

### 3. Batch Experiments with Multiple Seeds

Run experiments with multiple random seeds for robust evaluation:

```bash
# Multiple tasks with multiple seeds
python run_experiments.py \
    --tasks mnli qnli rte \
    --seeds 42 123 456 789 \
    --model_name_or_path bert-base-multilingual-cased \
    --num_train_epochs 3 \
    --output_dir ./experiment_results

# All tasks with multiple seeds
python run_experiments.py \
    --all_tasks \
    --seeds 42 123 456 \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir ./experiment_results
```

#### Arguments

- `--tasks`: List of tasks to run
- `--all_tasks`: Run all ViGLUE tasks
- `--seeds`: List of random seeds
- `--model_name_or_path`: Model to evaluate
- `--num_train_epochs`: Number of epochs per run
- `--output_dir`: Base output directory

## Examples

### Quick Test

Test the framework with reduced settings:

```bash
python run_viglue.py \
    --task vsfc \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --quick_test \
    --max_train_samples 100 \
    --max_eval_samples 50
```

### Fine-tune Your Own BERT Model

If you have a custom BERT model (e.g., from previous conversations):

```bash
python run_viglue.py \
    --task vnrte \
    --model_name_or_path /path/to/your/bert/model \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --output_dir ./outputs/vnrte_custom
```

### Compare Multiple Models

Compare baseline vs fine-tuned performance:

```bash
# 1. Baseline
python run_baseline.py \
    --task mnli \
    --model_name_or_path bert-base-multilingual-cased \
    --output_dir ./results/baseline

# 2. Fine-tuned
python run_viglue.py \
    --task mnli \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --output_dir ./results/finetuned
```

### Robust Evaluation with Multiple Seeds

Get statistically robust results:

```bash
python run_experiments.py \
    --tasks mnli vnrte vsfc \
    --seeds 42 123 456 789 2023 \
    --model_name_or_path bert-base-multilingual-cased \
    --num_train_epochs 4 \
    --per_device_train_batch_size 32 \
    --output_dir ./robust_results
```

## Dataset Information

The framework uses the ViGLUE dataset from HuggingFace: [`tmnam20/ViGLUE`](https://huggingface.co/datasets/tmnam20/ViGLUE)

### Task Statistics

| Corpus | Train | Validation | Test | Metric | Domain |
|--------|-------|------------|------|--------|--------|
| MNLI | 392,702 | 9,815 | 9,796 | Acc. | Miscellaneous |
| QNLI | 104,743 | 5,463 | 5,463 | Acc. | Wikipedia |
| RTE | 2,490 | 277 | 3,000 | Acc. | Miscellaneous |
| VNRTE | 12,526 | 3,137 | - | Acc. | News |
| WNLI | 635 | 71 | 146 | Acc. | Fiction books |
| SST2 | 67,349 | 872 | 1,821 | Acc. | Movie reviews |
| VSFC | 11,426 | 1,538 | 3,166 | Acc. | Student feedback |
| VSMEC | 5,548 | 686 | 693 | Acc. | Social media |
| MRPC | 3,668 | 408 | 1,725 | Acc./F1 | News |
| QQP | 363,846 | 40,430 | 390,965 | Acc./F1 | Quora QA |
| CoLA | 8,551 | 1,043 | 1,063 | MCC | Miscellaneous |
| VToC | 7,293 | 1,831 | - | Acc. | News |

## Output Files

Each run creates an output directory with:

- `final_results.json`: Main evaluation results
- `train_metrics.json`: Training metrics (if `--do_train`)
- `training_config.json`: Training configuration
- `args.json`: Command-line arguments
- `train.log`: Training logs
- `checkpoint-*/`: Model checkpoints

For batch experiments:
- `all_experiments.json`: All individual and aggregated results
- `summary.txt`: Human-readable summary with mean ± std

## References

- **ViGLUE Dataset**: [HuggingFace](https://huggingface.co/datasets/tmnam20/ViGLUE) | [GitHub](https://github.com/trminhnam/ViGLUE)
- **Paper**: [ViGLUE: A Vietnamese General Language Understanding Evaluation Benchmark](https://github.com/trminhnam/ViGLUE)