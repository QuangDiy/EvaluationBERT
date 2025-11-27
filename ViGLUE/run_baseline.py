"""
Script for baseline evaluation (zero-shot) of BERT models on ViGLUE tasks.

This script evaluates a pre-trained model WITHOUT fine-tuning to establish baseline performance.

Usage:
    python run_baseline.py --task mnli --model_name_or_path bert-base-multilingual-cased
    python run_baseline.py --all_tasks --model_name_or_path bert-base-multilingual-cased
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
from typing import Dict
from tqdm import tqdm

from config.task_config import get_task_config, get_all_task_names
from utils.data_loader import load_and_prepare_dataset
from utils.model_utils import load_model_and_tokenizer
from utils.logging_utils import setup_logging, save_results, print_results, create_output_dir
from trainers.evaluator import ViGLUEEvaluator

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline evaluation of BERT models on ViGLUE (no fine-tuning)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=get_all_task_names(),
        help="ViGLUE task name (use --all_tasks to run all)"
    )
    parser.add_argument(
        "--all_tasks",
        action="store_true",
        help="Evaluate on all ViGLUE tasks"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-multilingual-cased",
        help="Path to pretrained model or model identifier"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["validation", "test", "all"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for datasets and models"
    )
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not args.task and not args.all_tasks:
        parser.error("Either --task or --all_tasks must be specified")
    
    if args.task and args.all_tasks:
        parser.error("Cannot specify both --task and --all_tasks")
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_single_task(
    task_name: str,
    model_name_or_path: str,
    args,
) -> Dict[str, float]:
    """
    Evaluate model on a single task.
    
    Args:
        task_name: Name of the task
        model_name_or_path: Model name or path
        args: Command-line arguments
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating task: {task_name}")
    logger.info(f"{'=' * 60}")
    
    task_config = get_task_config(task_name)
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        task_config=task_config,
        cache_dir=args.cache_dir,
    )
    
    logger.info("Loading dataset...")
    dataset = load_and_prepare_dataset(
        task_config=task_config,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    
    evaluator = ViGLUEEvaluator(
        model=model,
        tokenizer=tokenizer,
        task_config=task_config,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    results = {}
    if args.eval_split == "all":
        results = evaluator.evaluate_all_splits(dataset)
    else:
        metrics = evaluator.evaluate(dataset, split=args.eval_split)
        results[args.eval_split] = metrics
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "baseline_evaluation.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("ViGLUE Baseline Evaluation (No Fine-tuning)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.all_tasks:
        tasks = get_all_task_names()
        logger.info(f"Evaluating all {len(tasks)} tasks")
    else:
        tasks = [args.task]
        logger.info(f"Evaluating single task: {args.task}")
    
    all_results = {}
    for task_name in tqdm(tasks, desc="Evaluating tasks", unit="task"):
        try:
            results = evaluate_single_task(
                task_name=task_name,
                model_name_or_path=args.model_name_or_path,
                args=args,
            )
            all_results[task_name] = results
            
            print_results(results, f"Results for {task_name}")
            
            # Save individual task results in the main output directory
            save_results(results, args.output_dir, "baseline_results.json")
            
        except Exception as e:
            logger.error(f"Error evaluating task {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}
    
    save_results(all_results, args.output_dir, "all_baseline_results.json")
    
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE EVALUATION SUMMARY")
    logger.info("=" * 60)
    
    for task_name, results in all_results.items():
        if "error" in results:
            logger.info(f"{task_name:15s}: ERROR - {results['error']}")
        else:
            first_split = list(results.keys())[0]
            if isinstance(results[first_split], dict):
                first_metric = list(results[first_split].keys())[0]
                value = results[first_split][first_metric]
                logger.info(f"{task_name:15s}: {first_metric}={value:.4f}")
    
    logger.info("=" * 60)
    logger.info("Baseline evaluation completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
