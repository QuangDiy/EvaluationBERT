import logging
import os
from typing import Optional, Dict, Any
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from datasets import DatasetDict

from config.task_config import TaskConfig
from config.training_config import TrainingConfig
from utils.metrics import create_compute_metrics_fn, get_primary_metric

logger = logging.getLogger(__name__)


class ViGLUETrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: DatasetDict,
        task_config: TaskConfig,
        training_config: TrainingConfig,
    ):
        """
        Initialize ViGLUE trainer.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            dataset: DatasetDict with train/validation/test splits
            task_config: Task-specific configuration
            training_config: Training hyperparameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.task_config = task_config
        self.training_config = training_config
        
        self.training_args = self._create_training_arguments()
        
        self.compute_metrics_fn = create_compute_metrics_fn(task_config)
        
        self.trainer = self._create_trainer()
    
    def _create_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments from config."""
        primary_metric = get_primary_metric(self.task_config)
        metric_for_best_model = f"eval_{primary_metric}"
        
        save_strategy = self.training_config.save_strategy
        load_best_model_at_end = self.training_config.load_best_model_at_end
        eval_strategy = self.training_config.evaluation_strategy
        
        if not self.training_config.save_model:
            save_strategy = "no"
            load_best_model_at_end = False
        
        if self.task_config.use_validation_as_test:
            logger.info(
                f"Task {self.task_config.name} uses validation as test, "
                f"disabling mid-training evaluation"
            )
            eval_strategy = "no"
            load_best_model_at_end = False
        
        save_safetensors = True
        save_on_each_node = False
        
        if not self.training_config.save_model:
            save_safetensors = False
            save_on_each_node = False
        
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            overwrite_output_dir=self.training_config.overwrite_output_dir,
            
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            
            weight_decay=self.training_config.weight_decay,
            adam_epsilon=self.training_config.adam_epsilon,
            max_grad_norm=self.training_config.max_grad_norm,
            warmup_ratio=self.training_config.warmup_ratio,
            warmup_steps=self.training_config.warmup_steps,
            
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            logging_steps=self.training_config.logging_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            
            fp16=self.training_config.fp16,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory,
            
            logging_dir=self.training_config.logging_dir,
            seed=self.training_config.seed,
            no_cuda=self.training_config.no_cuda,
            local_rank=self.training_config.local_rank,
            report_to=self.training_config.report_to,
            
            save_safetensors=save_safetensors,
            save_on_each_node=save_on_each_node,
        )
        
        return training_args
    
    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer."""
        callbacks = []
        # Only add early stopping if we're doing evaluation during training
        if (self.training_config.early_stopping_patience > 0 and 
            self.training_args.eval_strategy != "no"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold,
                )
            )
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset.get("train"),
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_fn,
            callbacks=callbacks,
        )
        
        return trainer
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting training for task: {self.task_config.name}")
        
        train_result = self.trainer.train()
        
        if self.training_config.save_model:
            self.trainer.save_model()
        else:
            import shutil
            output_dir = self.training_config.output_dir
            if os.path.exists(output_dir):
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path) and item.startswith('checkpoint-'):
                        logger.info(f"Removing checkpoint directory: {item_path}")
                        shutil.rmtree(item_path)
                    elif item in ['pytorch_model.bin', 'model.safetensors', 'config.json', 
                                  'training_args.bin', 'optimizer.pt', 'scheduler.pt',
                                  'trainer_state.json', 'rng_state.pth']:
                        logger.info(f"Removing model file: {item_path}")
                        os.remove(item_path)
        
        metrics = train_result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return metrics
    
    def evaluate(self, dataset_split: str = "validation") -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataset_split: Which split to evaluate on ("validation" or "test")
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_split} set")
        
        eval_dataset = self.dataset.get(dataset_split)
        if eval_dataset is None:
            raise ValueError(f"Dataset split '{dataset_split}' not found")
        
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def predict(self, dataset_split: str = "test") -> Dict[str, Any]:
        """
        Generate predictions.
        
        Args:
            dataset_split: Which split to predict on
            
        Returns:
            Prediction results
        """
        logger.info(f"Generating predictions for {dataset_split} set")
        
        predict_dataset = self.dataset.get(dataset_split)
        if predict_dataset is None:
            raise ValueError(f"Dataset split '{dataset_split}' not found")
        
        predictions = self.trainer.predict(predict_dataset)
        
        return {
            "predictions": predictions.predictions,
            "label_ids": predictions.label_ids,
            "metrics": predictions.metrics,
        }
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save model and tokenizer."""
        if output_dir is None:
            output_dir = self.training_config.output_dir
        
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
