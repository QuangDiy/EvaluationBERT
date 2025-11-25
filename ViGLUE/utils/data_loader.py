import logging
from typing import Dict, Optional, Callable
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from config.task_config import TaskConfig

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    task_config: TaskConfig,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 128,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load and preprocess dataset for a specific task.
    
    Args:
        task_config: Configuration for the task
        tokenizer: Tokenizer for preprocessing
        max_seq_length: Maximum sequence length
        cache_dir: Directory for caching datasets
        
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    logger.info(f"Loading dataset: {task_config.dataset_name} ({task_config.dataset_config})")
    
    dataset = load_dataset(
        task_config.dataset_name,
        task_config.dataset_config,
        cache_dir=cache_dir,
    )
    
    preprocess_fn = create_preprocess_function(
        task_config=task_config,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    
    logger.info("Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        desc=f"Preprocessing {task_config.name}",
    )
    
    columns_to_remove = [
        col for col in processed_dataset["train"].column_names
        if col not in ["input_ids", "attention_mask", "token_type_ids", "labels"]
    ]
    processed_dataset = processed_dataset.remove_columns(columns_to_remove)
    
    logger.info(f"Dataset loaded: {processed_dataset}")
    return processed_dataset


def create_preprocess_function(
    task_config: TaskConfig,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 128,
) -> Callable:
    """
    Create preprocessing function for a specific task.
    
    Args:
        task_config: Configuration for the task
        tokenizer: Tokenizer for preprocessing
        max_seq_length: Maximum sequence length
        
    Returns:
        Preprocessing function
    """
    input_columns = task_config.input_columns
    
    def preprocess_function(examples: Dict) -> Dict:
        """Preprocess a batch of examples."""
        if len(input_columns) == 1:
            texts = examples[input_columns[0]]
            encoded = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
            )
        else:
            texts_a = examples[input_columns[0]]
            texts_b = examples[input_columns[1]]
            encoded = tokenizer(
                texts_a,
                texts_b,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
            )
        
        if "label" in examples:
            encoded["labels"] = examples["label"]
        
        return encoded
    
    return preprocess_function


def preprocess_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    input_columns: tuple,
    max_seq_length: int = 128,
) -> Dict:
    """
    Preprocess examples for any task (legacy function).
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer for preprocessing
        input_columns: Tuple of input column names
        max_seq_length: Maximum sequence length
        
    Returns:
        Preprocessed batch
    """
    if len(input_columns) == 1:
        texts = examples[input_columns[0]]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
    else:
        texts_a = examples[input_columns[0]]
        texts_b = examples[input_columns[1]]
        encoded = tokenizer(
            texts_a,
            texts_b,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )
    
    if "label" in examples:
        encoded["labels"] = examples["label"]
    
    return encoded


def get_dataset_info(dataset: DatasetDict) -> Dict:
    """
    Get information about dataset splits.
    
    Args:
        dataset: DatasetDict object
        
    Returns:
        Dictionary with split information
    """
    info = {}
    for split_name in dataset.keys():
        info[split_name] = {
            "num_examples": len(dataset[split_name]),
            "features": list(dataset[split_name].features.keys()),
        }
    return info


def sample_dataset(dataset: Dataset, num_samples: int, seed: int = 42) -> Dataset:
    """
    Sample a subset of the dataset for quick testing.
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to select
        seed: Random seed
        
    Returns:
        Sampled dataset
    """
    if len(dataset) <= num_samples:
        return dataset
    
    return dataset.shuffle(seed=seed).select(range(num_samples))
