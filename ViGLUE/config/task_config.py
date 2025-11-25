from typing import Dict, Optional, Tuple, List


class TaskConfig:
    """Configuration for a single task in ViGLUE."""
    
    def __init__(
        self,
        name: str,
        dataset_name: str,
        dataset_config: Optional[str],
        input_columns: Tuple[str, ...],
        num_labels: int,
        task_type: str,
        metrics: List[str],
        is_regression: bool = False,
    ):
        self.name = name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.input_columns = input_columns
        self.num_labels = num_labels
        self.task_type = task_type
        self.metrics = metrics
        self.is_regression = is_regression


TASK_CONFIG = {
    # Natural Language Inference Tasks
    "mnli": TaskConfig(
        name="mnli",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="mnli",
        input_columns=("premise", "hypothesis"),
        num_labels=3,  # entailment, neutral, contradiction
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
    ),
    "qnli": TaskConfig(
        name="qnli",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="qnli",
        input_columns=("question", "sentence"),
        num_labels=2,  # entailment, not_entailment
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
    ),
    "rte": TaskConfig(
        name="rte",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="rte",
        input_columns=("sentence1", "sentence2"),
        num_labels=2,  # entailment, not_entailment
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
    ),
    "vnrte": TaskConfig(
        name="vnrte",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="vnrte",
        input_columns=("sentence1", "sentence2"),
        num_labels=2,  # entailment, not_entailment
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
    ),
    "wnli": TaskConfig(
        name="wnli",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="wnli",
        input_columns=("sentence1", "sentence2"),
        num_labels=2,  # entailment, not_entailment
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
    ),
    
    # Sentiment Analysis Tasks
    "sst2": TaskConfig(
        name="sst2",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="sst2",
        input_columns=("sentence",),
        num_labels=2,  # positive, negative
        task_type="sentiment",
        metrics=["accuracy"],
        is_regression=False,
    ),
    "vsfc": TaskConfig(
        name="vsfc",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="vsfc",
        input_columns=("sentence",),
        num_labels=3,  # positive, neutral, negative
        task_type="sentiment",
        metrics=["accuracy", "f1"],
        is_regression=False,
    ),
    "vsmec": TaskConfig(
        name="vsmec",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="vsmec",
        input_columns=("sentence",),
        num_labels=7,  # joy, sadness, anger, fear, surprise, disgust, other
        task_type="sentiment",
        metrics=["accuracy"],
        is_regression=False,
    ),
    
    # Similarity and Paraphrase Tasks
    "mrpc": TaskConfig(
        name="mrpc",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="mrpc",
        input_columns=("sentence1", "sentence2"),
        num_labels=2,  # equivalent, not_equivalent
        task_type="similarity",
        metrics=["accuracy", "f1"],
        is_regression=False,
    ),
    "qqp": TaskConfig(
        name="qqp",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="qqp",
        input_columns=("question1", "question2"),
        num_labels=2,  # duplicate, not_duplicate
        task_type="similarity",
        metrics=["accuracy", "f1"],
        is_regression=False,
    ),
    "stsb": TaskConfig(
        name="stsb",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="stsb",
        input_columns=("sentence1", "sentence2"),
        num_labels=1,  # similarity score 0-5
        task_type="similarity",
        metrics=["pearson", "spearmanr"],
        is_regression=True,
    ),
    
    # Single-Sentence Tasks
    "cola": TaskConfig(
        name="cola",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="cola",
        input_columns=("sentence",),
        num_labels=2,  # acceptable, unacceptable
        task_type="acceptability",
        metrics=["matthews_correlation"],
        is_regression=False,
    ),
    "vtoc": TaskConfig(
        name="vtoc",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="vtoc",
        input_columns=("sentence",),
        num_labels=10,  # 10 topic categories
        task_type="classification",
        metrics=["accuracy"],
        is_regression=False,
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """
    Get configuration for a specific task.
    
    Args:
        task_name: Name of the task (e.g., 'mnli', 'qnli', etc.)
        
    Returns:
        TaskConfig object for the specified task
        
    Raises:
        ValueError: If task_name is not recognized
    """
    task_name = task_name.lower()
    if task_name not in TASK_CONFIG:
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {list(TASK_CONFIG.keys())}"
        )
    return TASK_CONFIG[task_name]


def get_all_task_names() -> List[str]:
    """Get list of all available task names."""
    return list(TASK_CONFIG.keys())
