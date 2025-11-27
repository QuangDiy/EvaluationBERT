"""
Generate GLUE submission files from trained models.

This script generates TSV files in GLUE submission format for uploading to the GLUE benchmark.
It handles special cases like MNLI matched/mismatched splits and Vietnamese tasks.
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))

from config.task_config import get_task_config, get_all_task_names
from utils.submission_utils import (
    generate_tsv,
    validate_submission,
    create_submission_zip,
    get_submission_files_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GLUE submission files")
    
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        type=str,
        help="Single task name to generate submission for",
    )
    task_group.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Multiple tasks to generate submissions for",
    )
    task_group.add_argument(
        "--all-glue",
        action="store_true",
        help="Generate submissions for all GLUE tasks",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to single trained model (required if --task is used)",
    )
    parser.add_argument(
        "--model-path-pattern",
        type=str,
        help="Pattern for model paths, use {task} placeholder (e.g., './results/{task}/best_model')",
    ) 
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./submissions",
        help="Output directory for TSV files (default: ./submissions)",
    )
    parser.add_argument(
        "--create-zip",
        action="store_true",
        help="Create submission.zip file",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="submission.zip",
        help="Name for submission zip file (default: submission.zip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    
    return parser.parse_args()


def get_tasks_to_process(args) -> List[str]:
    """Determine which tasks to process based on arguments."""
    if args.task:
        return [args.task]
    elif args.tasks:
        return args.tasks
    elif args.all_glue:
        glue_tasks = ["mnli", "qnli", "rte", "wnli", "sst2", "qqp", "cola", "mrpc", "stsb"]
        return glue_tasks
    else:
        raise ValueError("Must specify --task, --tasks, or --all-glue")


def get_model_path(task_name: str, args) -> str:
    """Get model path for a specific task."""
    if args.model_path:
        return args.model_path
    elif args.model_path_pattern:
        return args.model_path_pattern.format(task=task_name)
    else:
        raise ValueError("Must specify --model-path or --model-path-pattern")


def load_model_and_tokenizer(model_path: str, device: str, task_config=None):
    """Load model and tokenizer from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    config = AutoConfig.from_pretrained(model_path)
    
    if hasattr(config, 'classifier_dropout') and config.classifier_dropout is None:
        config.classifier_dropout = 0.1
        logger.info("Set classifier_dropout to 0.1 (was None)")
    
    if task_config:
        if task_config.is_regression:
            num_labels = 1
        else:
            num_labels = task_config.num_labels
        
        config.num_labels = num_labels
        logger.info(f"Loading model with {num_labels} labels for task {task_config.name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    dataset,
    task_config,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple:
    """
    Generate predictions for a dataset.
    
    Returns:
        Tuple of (predictions, indices)
    """
    logger.info(f"Generating predictions for {len(dataset)} examples")
    
    all_predictions = []
    all_indices = []
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), total=num_batches, desc="Generating predictions"):
            batch = dataset[i : i + batch_size]
            
            if "idx" in batch:
                indices = batch["idx"]
            elif "index" in batch:
                indices = batch["index"]
            else:
                indices = list(range(i, i + len(batch["label"])))
            
            all_indices.extend(indices)
            
            if len(task_config.input_columns) == 1:
                texts = batch[task_config.input_columns[0]]
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            else:
                texts1 = batch[task_config.input_columns[0]]
                texts2 = batch[task_config.input_columns[1]]
                inputs = tokenizer(
                    texts1,
                    texts2,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            if task_config.is_regression:
                predictions = logits.squeeze(-1).cpu().numpy()
            else:
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    return np.array(all_predictions), all_indices


def generate_submission_for_task(
    task_name: str,
    model_path: str,
    output_dir: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> List[str]:
    """
    Generate submission files for a single task.
    
    Returns:
        List of generated TSV file paths
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task_name}")
    logger.info(f"{'='*60}")
    
    task_config = get_task_config(task_name)
    
    if not task_config.submission_name:
        logger.warning(f"Task {task_name} has no submission_name, skipping")
        return []
    
    model, tokenizer = load_model_and_tokenizer(model_path, device, task_config)
    
    splits_mapping = get_submission_files_mapping(task_config)
    
    generated_files = []
    
    for split_name, file_suffix in splits_mapping.items():
        logger.info(f"Processing split: {split_name}")
        
        try:
            dataset = load_dataset(
                task_config.dataset_name,
                task_config.dataset_config,
                split=split_name,
            )
            
            logger.info(f"Loaded {len(dataset)} examples from {split_name}")
            
            predictions, indices = generate_predictions(
                model,
                tokenizer,
                dataset,
                task_config,
                batch_size,
                max_length,
                device,
            )
            
            tsv_path = generate_tsv(
                predictions,
                indices,
                task_config,
                output_dir,
                split_suffix=file_suffix,
            )
            
            if validate_submission(tsv_path):
                generated_files.append(tsv_path)
            else:
                logger.error(f"Validation failed for {tsv_path}")
                
        except Exception as e:
            logger.error(f"Error processing split {split_name}: {e}")
            continue
    
    return generated_files


def main():
    args = parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    tasks = get_tasks_to_process(args)
    logger.info(f"Processing {len(tasks)} tasks: {tasks}")
    
    all_generated_files = []
    
    for task_name in tasks:
        try:
            model_path = get_model_path(task_name, args)
            
            generated_files = generate_submission_for_task(
                task_name,
                model_path,
                args.output_dir,
                args.batch_size,
                args.max_length,
                device,
            )
            
            all_generated_files.extend(generated_files)
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    if args.create_zip and all_generated_files:
        zip_path = create_submission_zip(
            all_generated_files,
            args.output_dir,
            args.zip_name,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Submission zip created: {zip_path}")
        logger.info(f"{'='*60}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Generated {len(all_generated_files)} submission files:")
    for filepath in all_generated_files:
        logger.info(f"  - {os.path.basename(filepath)}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
