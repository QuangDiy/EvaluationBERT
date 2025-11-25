import os
import json
import logging
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    logger.handlers = []
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_dir(base_dir: str, task_name: str, seed: Optional[int] = None) -> str:
    """
    Create output directory for results.
    
    Args:
        base_dir: Base output directory
        task_name: Name of the task
        seed: Random seed (optional)
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if seed is not None:
        dir_name = f"{task_name}_seed{seed}_{timestamp}"
    else:
        dir_name = f"{task_name}_{timestamp}"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str = "results.json",
) -> str:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results to save
        output_dir: Output directory
        filename: Name of output file
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Results saved to {output_path}")
    return output_path


def save_metrics_to_csv(
    metrics: Dict[str, float],
    output_file: str,
    task_name: str,
    seed: Optional[int] = None,
    append: bool = True,
):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metric name -> value
        output_file: Path to CSV file
        task_name: Name of the task
        seed: Random seed (optional)
        append: Whether to append to existing file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    row = {"task": task_name}
    if seed is not None:
        row["seed"] = seed
    row.update(metrics)
    
    file_exists = os.path.isfile(output_file) and append
    
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    logging.info(f"Metrics saved to {output_file}")


def save_config(config: Any, output_dir: str, filename: str = "config.json"):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration object (must have to_dict method or be dict)
        output_dir: Output directory
        filename: Name of output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = vars(config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Configuration saved to {output_path}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def print_results(results: Dict[str, Any], title: str = "Results"):
    """
    Pretty print results.
    
    Args:
        results: Dictionary of results
        title: Title for the results
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"  {sub_key:28s}: {sub_value:.4f}")
                else:
                    print(f"  {sub_key:28s}: {sub_value}")
        else:
            print(f"{key:30s}: {value}")
    
    print(f"{'=' * 60}\n")
