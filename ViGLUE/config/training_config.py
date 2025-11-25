from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training and evaluation hyperparameters."""
    
    model_name_or_path: str = "bert-base-multilingual-cased"
    max_seq_length: int = 128
    
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    
    seed: int = 42
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 100
    eval_steps: int = 500
    evaluation_strategy: str = "steps"  # "no", "steps", "epoch"
    save_strategy: str = "steps"  # "no", "steps", "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    
    fp16: bool = False  # Mixed precision training
    fp16_opt_level: str = "O1"
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = False
    overwrite_output_dir: bool = True
    
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    no_cuda: bool = False
    local_rank: int = -1
    report_to: str = "none"  # "none", "tensorboard", "wandb"
    
    def __post_init__(self):
        """Set up derived configurations."""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_args(cls, args):
        """Create TrainingConfig from command-line arguments."""
        config_dict = {}
        for key in cls.__dataclass_fields__.keys():
            if hasattr(args, key):
                config_dict[key] = getattr(args, key)
        return cls(**config_dict)


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing (1 epoch, small batch)."""
    return TrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
    )


def get_robust_config() -> TrainingConfig:
    """Get configuration for robust evaluation (5 epochs, larger batch)."""
    return TrainingConfig(
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )
