import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr

from config.task_config import TaskConfig


def compute_metrics(eval_pred, task_config: TaskConfig) -> Dict[str, float]:
    """
    Compute metrics for a specific task.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        task_config: Configuration for the task
        
    Returns:
        Dictionary of metric name -> value
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    if not task_config.is_regression:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions.squeeze()
    else:
        predictions = predictions.squeeze()
    
    metrics = {}
    
    for metric_name in task_config.metrics:
        if metric_name == "accuracy":
            metrics["accuracy"] = accuracy_score(labels, predictions)
        
        elif metric_name == "f1":
            if task_config.num_labels == 2:
                metrics["f1"] = f1_score(labels, predictions, average="binary")
            else:
                metrics["f1"] = f1_score(labels, predictions, average="macro")
            
            if task_config.num_labels == 2:
                metrics["precision"] = precision_score(labels, predictions, average="binary")
                metrics["recall"] = recall_score(labels, predictions, average="binary")
            else:
                metrics["precision"] = precision_score(labels, predictions, average="macro")
                metrics["recall"] = recall_score(labels, predictions, average="macro")
        
        elif metric_name == "matthews_correlation":
            metrics["matthews_correlation"] = matthews_corrcoef(labels, predictions)
        
        elif metric_name == "pearson":
            pearson_corr, _ = pearsonr(labels, predictions)
            metrics["pearson"] = pearson_corr
        
        elif metric_name == "spearmanr":
            spearman_corr, _ = spearmanr(labels, predictions)
            metrics["spearmanr"] = spearman_corr
    
    return metrics


def create_compute_metrics_fn(task_config: TaskConfig):
    """
    Create a compute_metrics function for a specific task.
    
    Args:
        task_config: Configuration for the task
        
    Returns:
        Compute metrics function
    """
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, task_config)
    
    return compute_metrics_fn


def get_metric_names(task_config: TaskConfig) -> List[str]:
    """
    Get list of metric names for a task.
    
    Args:
        task_config: Configuration for the task
        
    Returns:
        List of metric names
    """
    metric_names = task_config.metrics.copy()
    
    if "f1" in metric_names:
        metric_names.extend(["precision", "recall"])
    
    return metric_names


def aggregate_metrics(metric_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple runs (e.g., different seeds).
    
    Args:
        metric_results: List of metric dictionaries from different runs
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metric_results:
        return {}
    
    aggregated = {}
    metric_names = metric_results[0].keys()
    
    for metric_name in metric_names:
        values = [result[metric_name] for result in metric_results]
        aggregated[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values,
        }
    
    return aggregated


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics as a readable string.
    
    Args:
        metrics: Dictionary of metric name -> value
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name}: {value:.{precision}f}")
    return "\n".join(lines)


def get_primary_metric(task_config: TaskConfig) -> str:
    """
    Get the primary metric for a task (used for model selection).
    
    Args:
        task_config: Configuration for the task
        
    Returns:
        Primary metric name
    """
    if task_config.metrics:
        return task_config.metrics[0]
    return "accuracy" 
