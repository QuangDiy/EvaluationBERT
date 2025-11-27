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
        submission_name: Optional[str] = None,
        test_splits: Optional[List[str]] = None,
        label_list: Optional[List[str]] = None,
        use_validation_as_test: bool = False,
    ):
        self.name = name
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.input_columns = input_columns
        self.num_labels = num_labels
        self.task_type = task_type
        self.metrics = metrics
        self.is_regression = is_regression
        self.submission_name = submission_name
        self.test_splits = test_splits or ["test"]
        self.label_list = label_list
        self.use_validation_as_test = use_validation_as_test


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
        submission_name="MNLI",
        test_splits=["test", "test_mismatched"],
        label_list=["entailment", "neutral", "contradiction"],
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
        submission_name="QNLI",
        label_list=["entailment", "not_entailment"],
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
        submission_name="RTE",
        label_list=["entailment", "not_entailment"],
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
        use_validation_as_test=True,
        label_list=["entailment", "not_entailment"],
    ),
    "wnli": TaskConfig(
        name="wnli",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="wnli",
        input_columns=("sentence1", "sentence2"),
        num_labels=2,  # not_entailment, entailment
        task_type="nli",
        metrics=["accuracy"],
        is_regression=False,
        submission_name="WNLI",
        label_list=["0", "1"],
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
        submission_name="SST-2",
        label_list=["0", "1"],  # 0=negative, 1=positive (GLUE format)
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
        submission_name="MRPC",
        label_list=["0", "1"],  # 0=not_equivalent, 1=equivalent (GLUE format)
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
        submission_name="QQP",
        label_list=["0", "1"],  # 0=not_duplicate, 1=duplicate (GLUE format)
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
        submission_name="STS-B",
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
        submission_name="CoLA",
        label_list=["0", "1"],  # 0=unacceptable, 1=acceptable (GLUE format)
    ),
    "vtoc": TaskConfig(
        name="vtoc",
        dataset_name="tmnam20/ViGLUE",
        dataset_config="vtoc",
        input_columns=("sentence",),
        num_labels=15, 
        task_type="classification",
        metrics=["accuracy"],
        is_regression=False,
        use_validation_as_test=True,
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
