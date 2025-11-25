import logging
import numpy as np
from typing import Dict, Optional, List
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict

from config.task_config import TaskConfig
from utils.metrics import create_compute_metrics_fn

logger = logging.getLogger(__name__)


class ViGLUEEvaluator:  
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task_config: TaskConfig,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            task_config: Task-specific configuration
            batch_size: Batch size for evaluation
            device: Device for evaluation (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.batch_size = batch_size
        self.device = device
        
        self.compute_metrics_fn = create_compute_metrics_fn(task_config)
    
    def evaluate(
        self,
        dataset: DatasetDict,
        split: str = "validation",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset split.
        
        Args:
            dataset: DatasetDict containing the data
            split: Split to evaluate on
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {split} split for task {self.task_config.name}")
        
        eval_dataset = dataset.get(split)
        if eval_dataset is None:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        training_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=self.batch_size,
            no_cuda=(self.device == "cpu") if self.device else False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_fn,
        )
        
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        clean_metrics = {
            k.replace("eval_", ""): v
            for k, v in metrics.items()
        }
        
        logger.info(f"Evaluation results: {clean_metrics}")
        return clean_metrics
    
    def evaluate_all_splits(
        self,
        dataset: DatasetDict,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all available splits.
        
        Args:
            dataset: DatasetDict containing all splits
            
        Returns:
            Dictionary mapping split name to metrics
        """
        results = {}
        
        for split in dataset.keys():
            logger.info(f"Evaluating on {split} split")
            try:
                metrics = self.evaluate(dataset, split=split)
                results[split] = metrics
            except Exception as e:
                logger.error(f"Error evaluating {split}: {e}")
                results[split] = {"error": str(e)}
        
        return results
    
    def predict(
        self,
        dataset: DatasetDict,
        split: str = "test",
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a dataset split.
        
        Args:
            dataset: DatasetDict containing the data
            split: Split to predict on
            
        Returns:
            Dictionary with predictions and labels
        """
        logger.info(f"Generating predictions for {split} split")
        
        predict_dataset = dataset.get(split)
        if predict_dataset is None:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        training_args = TrainingArguments(
            output_dir="./tmp_predict",
            per_device_eval_batch_size=self.batch_size,
            no_cuda=(self.device == "cpu") if self.device else False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_fn,
        )
        
        predictions = trainer.predict(predict_dataset)
        
        if not self.task_config.is_regression:
            if len(predictions.predictions.shape) > 1 and predictions.predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions.predictions, axis=1)
            else:
                pred_classes = predictions.predictions.squeeze()
        else:
            pred_classes = predictions.predictions.squeeze()
        
        return {
            "predictions": pred_classes,
            "logits": predictions.predictions,
            "labels": predictions.label_ids,
            "metrics": predictions.metrics,
        }
