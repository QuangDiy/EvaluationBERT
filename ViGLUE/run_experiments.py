"""
Script for running batch experiments on ViGLUE with multiple seeds.

This script allows running multiple experiments with different random seeds
to obtain robust evaluation metrics with mean and standard deviation.

Usage:
    python run_experiments.py --tasks mnli qnli --seeds 42 123 456 --model_name_or_path bert-base-multilingual-cased
    python run_experiments.py --all_tasks --seeds 42 123 456 789 --model_name_or_path bert-base-multilingual-cased
"""

import argparse
import logging
import os
import json
import subprocess
import sys
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from config.task_config import get_all_task_names
from utils.logging_utils import setup_logging, save_results, print_results
from utils.metrics import aggregate_metrics

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments on ViGLUE with multiple seeds"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of tasks to evaluate"
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
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="List of random seeds"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=64, 
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=128, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--do_train", 
        action="store_true", 
        default=True, 
        help="Whether to train"
    )
    parser.add_argument(
        "--do_eval", 
        action="store_true", 
        default=True, 
        help="Whether to evaluate"
    )
    parser.add_argument(
        "--quick_test", 
        action="store_true", 
        help="Quick test mode"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_results",
        help="Base output directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory"
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not save model checkpoints"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (experimental)"
    )
    
    args = parser.parse_args()
    
    if not args.tasks and not args.all_tasks:
        parser.error("Either --tasks or --all_tasks must be specified")
    
    return args


def run_single_experiment(
    task: str,
    seed: int,
    model_name_or_path: str,
    output_base_dir: str,
    args,
) -> Dict:
    """
    Run a single experiment for one task and one seed.
    
    Args:
        task: Task name
        seed: Random seed
        model_name_or_path: Model name or path
        output_base_dir: Base output directory
        args: Command-line arguments
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running experiment: task={task}, seed={seed}")
    logger.info(f"{'=' * 60}")
    
    exp_output_dir = os.path.join(output_base_dir, task, f"seed_{seed}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        "run_viglue.py",
        "--task", task,
        "--model_name_or_path", model_name_or_path,
        "--seed", str(seed),
        "--output_dir", exp_output_dir,
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length),
    ]
    
    if args.do_train:
        cmd.append("--do_train")
    if args.do_eval:
        cmd.append("--do_eval")
    if args.quick_test:
        cmd.append("--quick_test")
    if args.cache_dir:
        cmd.extend(["--cache_dir", args.cache_dir])
    if args.no_save_model:
        cmd.append("--no_save_model")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Experiment completed successfully")
        
        results_file = os.path.join(exp_output_dir, "final_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return {"status": "success", "results": results}
        else:
            logger.warning(f"Results file not found: {results_file}")
            return {"status": "no_results"}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return {"status": "error", "error": str(e)}


def aggregate_task_results(task_results: List[Dict]) -> Dict:
    """
    Aggregate results across multiple seeds for a task.
    
    Args:
        task_results: List of result dictionaries from different seeds
        
    Returns:
        Aggregated results with mean and std
    """
    successful_results = [
        r["results"] for r in task_results
        if r.get("status") == "success" and "results" in r
    ]
    
    if not successful_results:
        return {"error": "No successful results to aggregate"}
    
    aggregated = {}
    
    splits = set()
    for result in successful_results:
        splits.update(result.keys())
    
    for split in splits:
        split_metrics = [
            result[split] for result in successful_results
            if split in result
        ]
        
        if split_metrics:
            aggregated[split] = aggregate_metrics(split_metrics)
    
    return aggregated


def main():
    """Main function."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "experiments.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("ViGLUE Batch Experiments")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.all_tasks:
        tasks = get_all_task_names()
    else:
        tasks = args.tasks
    
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Total experiments: {len(tasks) * len(args.seeds)}")
    
    all_results = {}
    
    for task in tqdm(tasks, desc="Tasks", position=0):
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Task: {task}")
        logger.info(f"{'#' * 60}")
        
        task_results = []
        
        for seed in tqdm(args.seeds, desc=f"Seeds for {task}", position=1, leave=False):
            result = run_single_experiment(
                task=task,
                seed=seed,
                model_name_or_path=args.model_name_or_path,
                output_base_dir=args.output_dir,
                args=args,
            )
            task_results.append(result)
        
        aggregated = aggregate_task_results(task_results)
        all_results[task] = {
            "individual_runs": task_results,
            "aggregated": aggregated,
        }
        
        print_results(aggregated, f"Aggregated Results for {task}")
    
    save_results(all_results, args.output_dir, "all_experiments.json")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("ViGLUE Batch Experiments Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Tasks: {tasks}\n\n")
        
        for task, results in all_results.items():
            f.write(f"\nTask: {task}\n")
            f.write("-" * 60 + "\n")
            
            if "aggregated" in results and "validation" in results["aggregated"]:
                validation_agg = results["aggregated"]["validation"]
                
                for metric_name, metric_stats in validation_agg.items():
                    if isinstance(metric_stats, dict) and "mean" in metric_stats:
                        mean = metric_stats["mean"]
                        std = metric_stats["std"]
                        f.write(f"{metric_name:20s}: {mean:.4f} ± {std:.4f}\n")
                        logger.info(f"{task:15s} - {metric_name:15s}: {mean:.4f} ± {std:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
