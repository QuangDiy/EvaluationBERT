"""Utilities for generating GLUE submission files."""

import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from config.task_config import TaskConfig

logger = logging.getLogger(__name__)


def get_label_from_prediction(
    prediction: Union[int, float],
    task_config: TaskConfig,
) -> str:
    """
    Convert prediction to label string.
    
    Args:
        prediction: Model prediction (index for classification, float for regression)
        task_config: Task configuration
        
    Returns:
        Label string for submission
    """
    if task_config.is_regression:
        return str(float(prediction))
    
    if task_config.label_list is None:
        raise ValueError(f"Task {task_config.name} has no label_list defined")
    
    pred_idx = int(prediction)
    if pred_idx < 0 or pred_idx >= len(task_config.label_list):
        raise ValueError(
            f"Prediction {pred_idx} out of range for task {task_config.name} "
            f"with {len(task_config.label_list)} labels"
        )
    
    return task_config.label_list[pred_idx]


def generate_tsv(
    predictions: np.ndarray,
    indices: List[Union[int, str]],
    task_config: TaskConfig,
    output_path: str,
    split_suffix: str = "",
) -> str:
    """
    Generate TSV file from predictions.
    
    Args:
        predictions: Model predictions
        indices: Test example indices
        task_config: Task configuration
        output_path: Output directory
        split_suffix: Suffix for split (e.g., "-m" for matched, "-mm" for mismatched)
        
    Returns:
        Path to generated TSV file
    """
    if len(predictions) != len(indices):
        raise ValueError(
            f"Predictions ({len(predictions)}) and indices ({len(indices)}) "
            f"must have same length"
        )
    
    if task_config.submission_name:
        base_name = task_config.submission_name
    else:
        base_name = task_config.name.upper()
    
    filename = f"{base_name}{split_suffix}.tsv"
    filepath = os.path.join(output_path, filename)
    
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Writing submission to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("index\tprediction\n")
        
        for idx, pred in zip(indices, predictions):
            label = get_label_from_prediction(pred, task_config)
            f.write(f"{idx}\t{label}\n")
    
    logger.info(f"Generated {filename} with {len(predictions)} predictions")
    return filepath


def validate_submission(filepath: str) -> bool:
    """
    Validate TSV submission format.
    
    Args:
        filepath: Path to TSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            if len(lines) < 2:
                logger.error(f"{filepath}: File too short")
                return False
            
            header = lines[0].strip().split("\t")
            if len(header) != 2 or header[0] != "index" or header[1] != "prediction":
                logger.error(f"{filepath}: Invalid header: {header}")
                return False
            
            for i, line in enumerate(lines[1:], start=2):
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    logger.error(f"{filepath}:{i}: Expected 2 columns, got {len(parts)}")
                    return False
            
        logger.info(f"{filepath}: Valid submission format")
        return True
        
    except Exception as e:
        logger.error(f"Error validating {filepath}: {e}")
        return False


def create_submission_zip(
    tsv_files: List[str],
    output_path: str,
    zip_name: str = "submission.zip",
) -> str:
    """
    Create zip file of TSV submissions without subfolders.
    
    Args:
        tsv_files: List of TSV file paths
        output_path: Output directory for zip file
        zip_name: Name of zip file
        
    Returns:
        Path to created zip file
    """
    os.makedirs(output_path, exist_ok=True)
    zip_path = os.path.join(output_path, zip_name)
    
    logger.info(f"Creating submission zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for tsv_file in tsv_files:
            if not os.path.exists(tsv_file):
                logger.warning(f"File not found, skipping: {tsv_file}")
                continue
                
            arcname = os.path.basename(tsv_file)
            zf.write(tsv_file, arcname=arcname)
            logger.info(f"Added {arcname} to zip")
    
    logger.info(f"Created submission zip with {len(tsv_files)} files")
    return zip_path


def get_submission_files_mapping(task_config: TaskConfig) -> Dict[str, str]:
    """
    Get mapping of split names to submission file suffixes.
    
    Args:
        task_config: Task configuration
        
    Returns:
        Dictionary mapping split name to file suffix
    """
    if task_config.name == "mnli":
        return {
            "test": "-m",  # MNLI-m.tsv
            "test_mismatched": "-mm",  # MNLI-mm.tsv
        }
    else:
        return {split: "" for split in task_config.test_splits}
