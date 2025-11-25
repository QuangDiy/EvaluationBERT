import logging
import torch
from typing import Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from config.task_config import TaskConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str,
    task_config: TaskConfig,
    cache_dir: Optional[str] = None,
    from_scratch: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer for a specific task.
    
    Args:
        model_name_or_path: Path or name of pretrained model
        task_config: Configuration for the task
        cache_dir: Directory for caching models
        from_scratch: If True, initialize model from scratch (for baseline)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
    )
    
    logger.info(f"Loading model: {model_name_or_path}")
    
    if task_config.is_regression:
        num_labels = 1
        problem_type = "regression"
    else:
        num_labels = task_config.num_labels
        problem_type = "single_label_classification"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
        problem_type=problem_type,
        ignore_mismatched_sizes=True, 
    )
    
    logger.info(f"Model loaded with {num_labels} labels for {task_config.name}")
    return model, tokenizer


def initialize_model(
    model_name_or_path: str,
    num_labels: int,
    is_regression: bool = False,
    cache_dir: Optional[str] = None,
) -> PreTrainedModel:
    """
    Initialize a model for sequence classification.
    
    Args:
        model_name_or_path: Path or name of pretrained model
        num_labels: Number of labels for classification
        is_regression: Whether this is a regression task
        cache_dir: Directory for caching models
        
    Returns:
        Initialized model
    """
    problem_type = "regression" if is_regression else "single_label_classification"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels if not is_regression else 1,
        cache_dir=cache_dir,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )
    
    return model


def freeze_model_parameters(model: PreTrainedModel, freeze_embeddings: bool = True):
    """
    Freeze model parameters (for baseline evaluation).
    
    Args:
        model: Model to freeze
        freeze_embeddings: Whether to freeze embedding layers
    """
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    if freeze_embeddings and hasattr(model.base_model, 'embeddings'):
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
    
    logger.info("Model parameters frozen")


def count_parameters(model: PreTrainedModel) -> dict:
    """
    Count trainable and total parameters in model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
    }


def get_device() -> torch.device:
    """Get the device for model training/inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def move_model_to_device(model: PreTrainedModel, device: Optional[torch.device] = None):
    """
    Move model to specified device.
    
    Args:
        model: Model to move
        device: Target device (defaults to auto-detected)
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    logger.info(f"Model moved to {device}")
    return model
