"""
Main script for fine-tuning and evaluating BERT models on ViGLUE tasks.

Usage:
    python run_viglue.py --task mnli --model_name_or_path bert-base-multilingual-cased --do_train --do_eval
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
import torch

from config.task_config import get_task_config, get_all_task_names
from config.training_config import TrainingConfig
from utils.data_loader import load_and_prepare_dataset
from utils.model_utils import load_model_and_tokenizer, count_parameters
from utils.logging_utils import setup_logging, save_results, save_config, print_results, create_output_dir
from trainers.base_trainer import ViGLUETrainer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate BERT models on ViGLUE benchmark"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=get_all_task_names(),
        help="ViGLUE task name"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-multilingual-cased",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--do_train", 
        action="store_true",
        help="Whether to run training"
    )
    parser.add_argument(
        "--do_eval", 
        action="store_true", 
        help="Whether to run evaluation"
    )
    parser.add_argument(
        "--do_predict", 
        action="store_true", 
        help="Whether to run prediction"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", 
        type=int, 
        default=32, 
        help="Training batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", 
        type=int, 
        default=64, 
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=1, 
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.1, 
        help="Warmup ratio"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="Weight decay"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=128, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0, 
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs", 
        help="Output directory"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100, 
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500, 
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=500, 
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--evaluation_strategy", 
        type=str, 
        default="steps", 
        choices=["no", "steps", "epoch"]
    )
    parser.add_argument(
        "--save_strategy", 
        type=str, 
        default="steps", 
        choices=["no", "steps", "epoch"]
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=2, 
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--early_stopping_patience", 
        type=int, 
        default=3, 
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="Cache directory"
    )
    parser.add_argument(
        "--overwrite_output_dir", 
        action="store_true", 
        help="Overwrite output directory"
    )
    parser.add_argument(
        "--quick_test", 
        action="store_true", 
        help="Quick test with reduced settings"
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=None, 
        help="Maximum training samples (for testing)"
    )
    parser.add_argument(
        "--max_eval_samples", 
        type=int, 
        default=None, 
        help="Maximum evaluation samples (for testing)"
    )
    
    args = parser.parse_args()

    if not args.do_train and not args.do_eval and not args.do_predict:
        parser.error("At least one of --do_train, --do_eval, --do_predict must be set")
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function."""
    args = parse_args()
    
    set_seed(args.seed)
    
    output_dir = create_output_dir(args.output_dir, args.task, args.seed)
    args.output_dir = output_dir
    
    log_file = os.path.join(output_dir, "train.log")
    setup_logging(log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("ViGLUE Evaluation Framework")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")
    
    task_config = get_task_config(args.task)
    logger.info(f"Task type: {task_config.task_type}")
    logger.info(f"Number of labels: {task_config.num_labels}")
    logger.info(f"Metrics: {task_config.metrics}")
    
    training_config = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        cache_dir=args.cache_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    
    if args.quick_test:
        logger.info("Quick test mode enabled")
        training_config.num_train_epochs = 1
        training_config.save_steps = 100
        training_config.eval_steps = 100
        training_config.logging_steps = 50
    
    save_config(training_config, output_dir, "training_config.json")
    save_config(vars(args), output_dir, "args.json")
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        task_config=task_config,
        cache_dir=args.cache_dir,
    )
    
    param_info = count_parameters(model)
    logger.info(f"Total parameters: {param_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    
    logger.info("Loading dataset...")
    dataset = load_and_prepare_dataset(
        task_config=task_config,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    
    if args.max_train_samples and "train" in dataset:
        from utils.data_loader import sample_dataset
        dataset["train"] = sample_dataset(dataset["train"], args.max_train_samples, args.seed)
        logger.info(f"Sampled {args.max_train_samples} training examples")
    
    if args.max_eval_samples and "validation" in dataset:
        from utils.data_loader import sample_dataset
        dataset["validation"] = sample_dataset(dataset["validation"], args.max_eval_samples, args.seed)
        logger.info(f"Sampled {args.max_eval_samples} validation examples")
    
    logger.info("Creating trainer...")
    trainer = ViGLUETrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        task_config=task_config,
        training_config=training_config,
    )
    
    if args.do_train:
        logger.info("Starting training...")
        train_metrics = trainer.train()
        
        save_results(train_metrics, output_dir, "train_metrics.json")
        print_results(train_metrics, "Training Metrics")
    
    results = {}
    if args.do_eval:
        logger.info("Starting evaluation...")
        
        if "validation" in dataset:
            eval_metrics = trainer.evaluate(dataset_split="validation")
            results["validation"] = eval_metrics
            print_results(eval_metrics, "Validation Results")
        
        if "test" in dataset:
            test_metrics = trainer.evaluate(dataset_split="test")
            results["test"] = test_metrics
            print_results(test_metrics, "Test Results")

    if args.do_predict:
        logger.info("Generating predictions...")
        predictions = trainer.predict(dataset_split="test")
        
        np.save(os.path.join(output_dir, "predictions.npy"), predictions["predictions"])
        save_results(predictions["metrics"], output_dir, "prediction_metrics.json")
        print_results(predictions["metrics"], "Prediction Results")
    
    if results:
        save_results(results, output_dir, "final_results.json")
    
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
