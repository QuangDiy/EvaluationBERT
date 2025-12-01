# ViGLUE 
   
1. **Set up the environment**:
   ```bash
   git clone https://github.com/QuangDiy/EvaluationBERT.git
   cd d:\workspace\EvaluationBERT\ViGLUE
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
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

## 3. Train with Multiple GPUs

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    run_viglue.py \
    --task mnli \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir ./outputs/mnli
```

## References

- **ViGLUE Dataset**: [HuggingFace](https://huggingface.co/datasets/tmnam20/ViGLUE) | [GitHub](https://github.com/trminhnam/ViGLUE)
- **Paper**: [ViGLUE: A Vietnamese General Language Understanding Evaluation Benchmark](https://github.com/trminhnam/ViGLUE)