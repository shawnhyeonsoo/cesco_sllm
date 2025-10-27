#!/usr/bin/env python3
"""
Local training script for CESCO sLLM fine-tuning with LoRA.
Can be run on any machine with a GPU or even CPU (very slow).
"""
import os
import sys
import json
import random
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import Config


class GenerationCallback(TrainerCallback):
    """Callback to generate sample outputs during training."""
    
    def __init__(self, tokenizer, test_prompts, generate_every_n_steps=100):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generate_every_n_steps = generate_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate samples at regular intervals."""
        if state.global_step > 0 and state.global_step % self.generate_every_n_steps == 0:
            print("\n" + "=" * 80)
            print(f"GENERATION TEST at Step {state.global_step}")
            print("=" * 80)
            
            model.eval()
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"\n--- Sample {i} ---")
                #print(f"Prompt: {prompt[:200]}...")
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part (after the prompt)
                response = generated_text.strip()
                print(f"Generated: {response}\n")
            
            print("=" * 80 + "\n")
            model.train()
        
        return control


def setup_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False
):
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        model_name: Base model name
        use_4bit: Use 4-bit quantization (recommended for consumer GPUs)
        use_8bit: Use 8-bit quantization
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=Config.HUGGING_FACE_HUB_TOKEN,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config for consumer GPUs
    quantization_config = None
    if use_4bit:
        print("Using 4-bit quantization (reduces memory usage)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        token=Config.HUGGING_FACE_HUB_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not quantization_config else None,
    )
    
    # Prepare for training
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model, tokenizer


def setup_lora(model, lora_config: dict):
    """Setup LoRA configuration."""
    print("Setting up LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("lora_r", Config.LORA_R),
        lora_alpha=lora_config.get("lora_alpha", Config.LORA_ALPHA),
        lora_dropout=lora_config.get("lora_dropout", Config.LORA_DROPOUT),
        target_modules=lora_config.get("lora_target_modules", Config.LORA_TARGET_MODULES).split(","),
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def load_and_prepare_data(
    train_file: str,
    test_file: str,
    tokenizer,
    max_length: int = 2048
):
    """Load and tokenize the dataset."""
    print(f"Loading data from {train_file} and {test_file}")
    
    # Load datasets
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file
        }
    )
    
    def tokenize_function(examples):
        """Tokenize the text."""
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            # Full text = instruction + response
            # The model will learn to generate 'output' given 'instruction'
            full_text = f"{instruction}\n{output}"
            texts.append(full_text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # Create labels: mask the instruction part, only train on the output
        labels = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            # Tokenize instruction and output separately
            instruction_ids = tokenizer(
                f"{instruction}\n",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False
            )["input_ids"]
            
            full_ids = tokenizer(
                f"{instruction}\n{output}",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )["input_ids"]
            
            # Create labels: -100 for instruction (ignore), actual ids for output
            label = [-100] * len(instruction_ids) + full_ids[len(instruction_ids):]
            
            # Pad to max_length
            if len(label) < max_length:
                label = label + [-100] * (max_length - len(label))
            else:
                label = label[:max_length]
            
            labels.append(label)
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    print(f"Train samples: {len(tokenized_dataset['train'])}")
    print(f"Test samples: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str = "./lora_model",
    hyperparameters: dict = None,
    test_prompts: list = None,
    generate_every_n_steps: int = 100
):
    """Train the model with LoRA.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for model
        hyperparameters: Optional hyperparameters to override defaults
        test_prompts: List of test prompts to generate during training
        generate_every_n_steps: Generate samples every N steps
    """
    
    # Default hyperparameters
    hp = {
        "num_train_epochs": Config.NUM_EPOCHS,
        "per_device_train_batch_size": Config.BATCH_SIZE,
        "per_device_eval_batch_size": Config.BATCH_SIZE,
        "gradient_accumulation_steps": Config.GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": Config.LEARNING_RATE,
        "warmup_steps": Config.WARMUP_STEPS,
        "logging_steps": Config.LOGGING_STEPS,
        "save_steps": Config.SAVE_STEPS,
        "eval_steps": Config.EVAL_STEPS,
        "fp16": Config.FP16,
        "use_wandb": Config.USE_WANDB,
    }
    
    # Override with provided hyperparameters
    if hyperparameters:
        hp.update(hyperparameters)
    
    # Initialize W&B if enabled
    if hp["use_wandb"]:
        wandb.init(
            project=Config.WANDB_PROJECT,
            config=hp,
            name=f"local-training-{Path(output_dir).name}"
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        warmup_steps=hp["warmup_steps"],
        logging_steps=hp["logging_steps"],
        save_steps=hp["save_steps"],
        eval_steps=hp["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        fp16=hp["fp16"] and torch.cuda.is_available(),
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir=f"{output_dir}/logs",
        report_to="wandb" if hp["use_wandb"] else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Prepare generation callback if test prompts provided
    callbacks = []
    if test_prompts:
        print(f"Adding generation callback (will generate every {generate_every_n_steps} steps)")
        generation_callback = GenerationCallback(
            tokenizer=tokenizer,
            test_prompts=test_prompts,
            generate_every_n_steps=generate_every_n_steps
        )
        callbacks.append(generation_callback)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "model_name": Config.BASE_MODEL_NAME,
        "hyperparameters": hp,
        "training_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {output_dir}")
    
    if hp["use_wandb"]:
        wandb.finish()
    
    return trainer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Local training with LoRA")
    parser.add_argument("--train-file", type=str, default="data/train.jsonl",
                       help="Training data file (JSONL format)")
    parser.add_argument("--test-file", type=str, default="data/test.jsonl",
                       help="Test data file (JSONL format)")
    parser.add_argument("--output-dir", type=str, default="./lora_model",
                       help="Output directory for model")
    parser.add_argument("--model-name", type=str, default=Config.BASE_MODEL_NAME,
                       help="Base model name")
    parser.add_argument("--epochs", type=float, default=Config.NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=Config.LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=Config.MAX_SEQ_LENGTH,
                       help="Maximum sequence length")
    parser.add_argument("--use-4bit", action="store_true", default=True,
                       help="Use 4-bit quantization (default: True)")
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit",
                       help="Don't use quantization")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training (very slow)")
    parser.add_argument("--test-generation", action="store_true", default=True,
                       help="Generate test samples during training (default: True)")
    parser.add_argument("--no-test-generation", action="store_false", dest="test_generation",
                       help="Don't generate test samples during training")
    parser.add_argument("--generation-steps", type=int, default=5,
                       help="Generate samples every N steps (default: 50)")
    parser.add_argument("--num-test-samples", type=int, default=3,
                       help="Number of test samples to generate (default: 3)")
    
    args = parser.parse_args()
    
    # Check for GPU
    if not torch.cuda.is_available() and not args.cpu:
        print("⚠️  WARNING: No GPU detected!")
        print("\nOptions:")
        print("  1. Run on a machine with GPU (recommended)")
        print("  2. Use Google Colab (free GPU): https://colab.research.google.com")
        print("  3. Use AWS EC2 with GPU instance")
        print("  4. Force CPU training with --cpu flag (extremely slow)")
        print("\nFor Google Colab, upload this script and run:")
        print("  !python local_train.py")
        response = input("\nDo you want to continue with CPU? (yes/no): ")
        if response.lower() != "yes":
            sys.exit(0)
    
    # Display info
    print("=" * 80)
    print("LOCAL TRAINING WITH LORA")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max length: {args.max_length}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 80 + "\n")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=args.use_4bit and not args.cpu
    )
    
    # Setup LoRA
    model = setup_lora(model, {
        "lora_r": Config.LORA_R,
        "lora_alpha": Config.LORA_ALPHA,
        "lora_dropout": Config.LORA_DROPOUT,
        "lora_target_modules": Config.LORA_TARGET_MODULES,
    })
    
    # Load and prepare data
    dataset = load_and_prepare_data(
        args.train_file,
        args.test_file,
        tokenizer,
        args.max_length
    )
    
    # Prepare test prompts for generation during training
    test_prompts = []
    if args.test_generation:
        print(f"\nLoading {args.num_test_samples} test samples for generation during training...")
        with open(args.test_file, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        # Select a few diverse samples
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(test_data)), min(args.num_test_samples, len(test_data)))
        test_prompts = [test_data[i]["instruction"] for i in sample_indices]
        
        print(f"Selected {len(test_prompts)} test prompts for generation")
    
    # Train
    trainer = train_model(
        model,
        tokenizer,
        dataset["train"],
        dataset["test"],
        args.output_dir,
        hyperparameters={
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        test_prompts=test_prompts if args.test_generation else None,
        generate_every_n_steps=args.generation_steps
    )
    
    print("\n" + "=" * 80)
    print("✨ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"\nTo test the model:")
    print(f"  python test_inference.py --model-path {args.output_dir}")


if __name__ == "__main__":
    main()
