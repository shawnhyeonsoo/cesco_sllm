"""
Unsloth-based training script for SageMaker.
This script uses Unsloth for faster and more memory-efficient fine-tuning.
"""
import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path

# Install Unsloth if not available
try:
    import unsloth
except ImportError:
    print("Unsloth not found. Installing...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "unsloth @ git+https://github.com/unslothai/unsloth.git"
    ])
    print("Unsloth installed successfully!")

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import TrainerCallback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerationCallback(TrainerCallback):
    """Callback to generate sample outputs during training."""

    def __init__(self, tokenizer, test_prompts, 
                 test_responses,
                 generate_every_n_steps=100):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.test_responses = test_responses
        self.generate_every_n_steps = generate_every_n_steps
        logger.info(f"GenerationCallback initialized with {len(test_prompts)} test prompts")
        logger.info(f"Will generate every {generate_every_n_steps} steps")
    
    def alpaca_prompt(self, ) -> str:
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""
        return alpaca_prompt

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate samples at regular intervals."""
        if state.global_step > 0 and state.global_step % self.generate_every_n_steps == 0:
            logger.info("\n" + "=" * 80)
            logger.info(f"GENERATION TEST at Step {state.global_step}")
            logger.info("=" * 80)
            FastLanguageModel.for_inference(model)
            alpaca_prompt_template = self.alpaca_prompt()
            
            claim_accuracy = 0
            category_accuracy = 0

            print(len(self.test_prompts)) # 3
            print(len(self.test_responses)) # 3

            
            
            for i, prompt in enumerate(self.test_prompts):
                print("==PROMPT==")
                print(prompt)
                print("==GROUND TRUTH==")
                ground_truth = self.test_responses[i].replace(self.tokenizer.eos_token, "").strip()
                print(ground_truth)
                ground_truth = json.loads(ground_truth)

                alpaca_prompt = alpaca_prompt_template.format(
                    prompt.split("### Instruction:")[1].split("###")[0].strip(),
                    prompt.split("### Input:")[1].split("###")[0].strip(),
                    ""
                )

                gt_claim = ground_truth["claim_status"]
                gt_category = ground_truth["categories"]

                inputs = self.tokenizer(
                    alpaca_prompt,
                    return_tensors="pt",).to(model.device)
                outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
                generated_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generation_section = generated_output.split("### Response:")[-1].strip()
                print("==OUTPUT==")
                print(generation_section)
                try:
                    output = json.loads(generation_section)
                    print(json.dumps(output, ensure_ascii=False, indent=2))

                    pred_claim = output["claim_status"]
                    pred_category = output["categories"]
                    if pred_claim == gt_claim:
                        claim_accuracy += 1
                    if set(pred_category) == set(gt_category):
                        category_accuracy += 1
                except:
                    pass

            total_samples = len(self.test_prompts)
            claim_acc_percent = (claim_accuracy / total_samples) * 100
            category_acc_percent = (category_accuracy / total_samples) * 100
            logger.info(f"Claim Status Accuracy: {claim_accuracy}/{total_samples} ({claim_acc_percent:.2f}%)")
            logger.info(f"Categories Accuracy: {category_accuracy}/{total_samples} ({category_acc_percent:.2f}%)")
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "eval/claim_accuracy": claim_acc_percent,
                        "eval/category_accuracy": category_acc_percent,
                        "eval/claim_correct": claim_accuracy,
                        "eval/category_correct": category_accuracy,
                        "eval/total_samples": total_samples,
                        "step": state.global_step,
                    })
            except ImportError:
                pass
                
            print("==============================================")
            print(f"Claim Accuracy: {claim_accuracy}/{total_samples} ({claim_acc_percent:.2f}%)")
            print(f"Category Accuracy: {category_accuracy}/{total_samples} ({category_acc_percent:.2f}%)")
            wandb.log({
                "eval/claim_accuracy": claim_acc_percent,
                "eval/category_accuracy": category_acc_percent
            })
            """
            model.eval()
            for i, prompt in enumerate(self.test_prompts, 1):
                logger.info(f"\n--- Sample {i}/{len(self.test_prompts)} ---")
                
                # Show input preview
                if "### Input:" in prompt:
                    input_text = prompt.split("### Input:")[1].split("###")[0].strip()
                    logger.info(f"Input: {input_text[:100]}...")
                
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the full output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract ONLY the response part (after "### Response:")
                if "### Response:" in generated_text:
                    response = generated_text.split("### Response:")[-1].strip()
                else:
                    # Fallback: remove the prompt from the beginning
                    response = generated_text[len(prompt):].strip()
                
                logger.info(f"\nGenerated JSON Response:")
                logger.info("-" * 80)
                
                # Try to parse and pretty-print the JSON
                try:
                    # Try to extract JSON from the response
                    if "{" in response and "}" in response:
                        json_start = response.find("{")
                        json_end = response.rfind("}") + 1
                        json_str = response[json_start:json_end]
                        parsed = json.loads(json_str)
                        logger.info(json.dumps(parsed, ensure_ascii=False, indent=2))
                    else:
                        logger.info(response[:500])
                except json.JSONDecodeError:
                    logger.info(response[:500])
                
                logger.info("-" * 80)
                """

            model.train()
            logger.info("=" * 80 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    
    # SageMaker specific
    parser.add_argument("--train_dir", type=str, default="./data/unsloth_train_dataset.json")
    parser.add_argument("--test_dir", type=str, default="./data/unsloth_test_dataset.json")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./outputs"))
    
    # W&B
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="cesco-sllm-unsloth")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    
    # HuggingFace token
    parser.add_argument("--hf_token", type=str, default=None)
    
    return parser.parse_args()


def load_data(train_path: str, test_path: str = None):
    """
    Load training and optional test datasets from direct file paths.
    
    Args:
        train_path: Direct path to training JSON/JSONL file
        test_path: Direct path to test JSON/JSONL file (optional)
    """
    logger.info(f"Loading training data from {train_path}")
    
    # Load training dataset directly from file path
    if train_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files={'train': train_path}, split='train')
    else:
        dataset = load_dataset('json', data_files=train_path, split='train')

    logger.info(f"Loaded {len(dataset)} training examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # Load test dataset if provided
    test_dataset = None
    if test_path:
        logger.info(f"Loading test data from {test_path}")
        if test_path.endswith('.jsonl'):
            test_dataset = load_dataset('json', data_files={'test': test_path}, split='test')
        else:
            test_dataset = load_dataset('json', data_files=test_path, split='train')
        logger.info(f"Loaded {len(test_dataset)} test examples")
    
    return dataset, test_dataset


def format_prompts(examples, tokenizer):
    """
    Format examples into Alpaca-style prompts.
    Expects 'instruction', 'input', and 'output' fields.
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    EOS_TOKEN = tokenizer.eos_token
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise generation will go on forever!
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up W&B if requested
    if args.use_wandb:
        import wandb
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
        # Initialize wandb
        run_name = args.wandb_run_name or f"unsloth-{args.model_name.split('/')[-1]}-lr{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
            }
        )
        report_to = "wandb"
        logger.info(f"W&B initialized: project={args.wandb_project}, run={run_name}")
    else:
        report_to = "none"
        logger.info("W&B disabled")
    
    # Set up HuggingFace token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    logger.info("=" * 80)
    logger.info("UNSLOTH TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info("=" * 80)
    
    # Load model with Unsloth
    logger.info("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=args.load_in_4bit,
        token=args.hf_token,
    )
    logger.info("Model loaded successfully!")
    
    # Add LoRA adapters
    logger.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("LoRA adapters added!")
    
    # Load datasets
    train_dataset, test_dataset = load_data(args.train_dir, args.test_dir)
    
    # Format datasets
    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True,
    )
    
    if test_dataset:
        test_dataset = test_dataset.map(
            lambda examples: format_prompts(examples, tokenizer),
            batched=True,
        )
    
    # Create test prompts for generation callback
    logger.info("Creating test prompts for generation monitoring...")
    alpaca_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    
    # Get 3 random samples from training data for testing
    import random
    random.seed(42)
    test_indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))

    test_prompts = []
    test_responses = []
    for idx in test_indices:
        # Get original data (before formatting)
        sample = test_dataset[idx]
        # Extract instruction and input from the formatted text
        formatted_text = sample["text"]
        
        
        # Create prompt without the response part
        if "### Instruction:" in formatted_text and "### Input:" in formatted_text:
            instruction_start = formatted_text.find("### Instruction:") + len("### Instruction:")
            instruction_end = formatted_text.find("### Input:")
            instruction = formatted_text[instruction_start:instruction_end].strip()
            
            input_start = formatted_text.find("### Input:") + len("### Input:")
            input_end = formatted_text.find("### Response:")
            input_text = formatted_text[input_start:input_end].strip()
            
            prompt = alpaca_prompt_template.format(instruction, input_text)
            test_prompts.append(prompt)
            # Also store the ground truth response for accuracy checking
            response_start = formatted_text.find("### Response:") + len("### Response:")
            response_text = formatted_text[response_start:].strip()
            test_responses.append(response_text)
    
    logger.info(f"Created {len(test_prompts)} test prompts for monitoring")
    logger.info(f"Created {len(test_responses)} test responses for monitoring")
    
    print("==============================================")
    # Create generation callback
    generation_callback = GenerationCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        test_responses=test_responses,
        generate_every_n_steps=args.save_steps  # Generate at same intervals as saving
    )
    
    # Set up training configuration
    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to=report_to,
        save_steps=args.save_steps,
        save_strategy="steps",
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,  # Can make training 5x faster for short sequences
        args=training_args,
        callbacks=[generation_callback],  # Add generation callback
    )
    
    # Train!
    logger.info("Starting training...")
    logger.info("Model will generate sample outputs every {} steps".format(args.save_steps))
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.model_dir}...")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Also save merged model (LoRA weights merged with base model)
    logger.info("Saving merged model...")
    model.save_pretrained_merged(
        f"{args.model_dir}/merged_model",
        tokenizer,
        save_method="merged_16bit",
    )
    
    # Save as GGUF for inference (optional)
    # model.save_pretrained_gguf(f"{args.model_dir}/gguf", tokenizer)
    
    logger.info("Training complete!")
    
    # Finish wandb run
    if args.use_wandb:
        import wandb
        wandb.finish()
        logger.info("W&B run finished")


if __name__ == "__main__":
    main()
