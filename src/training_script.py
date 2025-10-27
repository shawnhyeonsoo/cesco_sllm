"""
Training script for SageMaker fine-tuning with LoRA.
This script will be executed on the SageMaker training instance.
"""
import os
import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse training arguments."""
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Pretrained model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj",
                        help="Comma-separated list of target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every X updates steps")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use FP16 training")
    
    # SageMaker specific paths
    parser.add_argument("--train_data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test_data", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    
    # W&B logging
    parser.add_argument("--use_wandb", type=bool, default=True,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="cesco-sllm-finetuning",
                        help="W&B project name")
    
    args = parser.parse_args()
    return args


def preprocess_function(examples, tokenizer, max_seq_length):
    """Preprocess the dataset for instruction fine-tuning."""
    # Combine instruction and output
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"{instruction}\n{output}</s>"
        texts.append(text)
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs


def main():
    args = parse_args()
    
    logger.info(f"Starting training with arguments: {args}")
    
    # Initialize W&B if enabled
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
        token=hf_token
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    logger.info(f"Loading model from {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Loading datasets")
    train_files = [str(p) for p in Path(args.train_data).glob("*.jsonl")]
    test_files = [str(p) for p in Path(args.test_data).glob("*.jsonl")]
    
    logger.info(f"Train files: {train_files}")
    logger.info(f"Test files: {test_files}")
    
    train_dataset = load_dataset("json", data_files=train_files, split="train")
    eval_dataset = load_dataset("json", data_files=test_files, split="train")
    
    # Preprocess datasets
    logger.info("Preprocessing datasets")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",  # Updated from evaluation_strategy
        save_strategy="steps",
        fp16=args.fp16,
        report_to="wandb" if args.use_wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    logger.info(f"Final evaluation metrics: {metrics}")
    
    metrics_file = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
