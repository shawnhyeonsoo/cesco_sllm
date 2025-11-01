"""
Inference script to load the best saved model and run predictions on test data.
This script loads the merged best model (16-bit) for faster inference.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with best saved model"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name (must match the model used in training)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./outputs/best_model",
        help="Path to the best model checkpoint directory (with LoRA adapters)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/unsloth_test_dataset.json",
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./inference_results.json",
        help="Path to save inference results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--load_in_4bit",
        type=bool,
        default=True,
        help="Load model in 4-bit quantization"
    )
    return parser.parse_args()


def load_model(base_model_name: str, adapter_path: str, max_seq_length: int = 8096, load_in_4bit: bool = True):
    """Load the best saved model using Unsloth.
    
    Args:
        base_model_name: Name of the base model (e.g., "Qwen/Qwen2.5-3B-Instruct")
        adapter_path: Path to the LoRA adapter checkpoint
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit quantization
    """
    try:
        from unsloth import FastLanguageModel
        
        logger.info(f"Loading base model: {base_model_name}")
        logger.info(f"Loading LoRA adapters from: {adapter_path}")
        
        # Check if adapter path exists
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}\n"
                f"Expected structure:\n"
                f"  {adapter_path}/\n"
                f"    ├── adapter_config.json\n"
                f"    ├── adapter_model.safetensors\n"
                f"    └── tokenizer files..."
            )
        
        # First, load the base model
        logger.info("Step 1: Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect best dtype
            load_in_4bit=load_in_4bit,
        )
        
        # Then load the LoRA adapters from the checkpoint
        logger.info("Step 2: Loading LoRA adapters from best checkpoint...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Must match training config
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,  # Must match training config
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Load the saved adapter weights
        from peft import PeftModel
        logger.info("Step 3: Loading adapter weights from checkpoint...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Set model to inference mode
        FastLanguageModel.for_inference(model)
        
        logger.info("✅ Model loaded successfully with LoRA adapters!")
        logger.info(f"Model device: {model.device}")
        
        return model, tokenizer
        
    except ImportError:
        logger.error("Unsloth not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "unsloth @ git+https://github.com/unslothai/unsloth.git"
        ])
        logger.info("Unsloth installed. Please run the script again.")
        exit(1)


def load_test_data(test_path: str, max_samples: int = None):
    """Load test dataset."""
    logger.info(f"Loading test data from {test_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    # Load dataset
    if test_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=test_path, split="train")
    else:
        dataset = load_dataset("json", data_files=test_path, split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} test examples")
    return dataset


def create_prompt(instruction: str, input_text: str) -> str:
    """Create Alpaca-style prompt for inference."""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    return alpaca_prompt.format(instruction, input_text)


def run_inference(model, tokenizer, dataset, output_file: str):
    """Run inference on the dataset and save results."""
    results = []
    
    claim_correct = 0
    category_correct = 0
    major_category_correct = 0
    mid_category_correct = 0
    minor_category_correct = 0
    total_claims = 0
    total_samples = 0
    
    logger.info("Starting inference...")
    
    for idx, sample in enumerate(tqdm(dataset, desc="Running inference")):
        try:
            # Extract instruction and input
            instruction = sample["instruction"]
            input_text = sample["input"]
            ground_truth_str = sample["output"]
            
            # Parse ground truth
            try:
                ground_truth = json.loads(ground_truth_str)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse ground truth for sample {idx}")
                ground_truth = None
            
            # Create prompt
            prompt = create_prompt(instruction, input_text)
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.7,
                do_sample=False,  # Use greedy decoding for deterministic results
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print(outputs)
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated text for sample {idx}:\n{generated_text}\n")
            # Extract response (after "### Response:")
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            # Try to parse JSON response
            try:
                prediction = json.loads(response)
                is_valid_json = True
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON for sample {idx}: {response[:100]}")
                prediction = {"error": "invalid_json", "raw_response": response}
                is_valid_json = False
            
            # Calculate accuracy if we have ground truth and valid prediction
            if ground_truth and is_valid_json and "is_claim" in prediction:
                total_samples += 1
                
                # Claim accuracy
                if prediction["is_claim"] == ground_truth["is_claim"]:
                    claim_correct += 1
                
                # Category accuracy (only for claims)
                if ground_truth["is_claim"] == "claim":
                    total_claims += 1
                    
                    if "categories" in prediction and "categories" in ground_truth:
                        gt_categories = ground_truth["categories"]
                        pred_categories = prediction["categories"]
                        
                        # Extract category levels
                        gt_major = [cat.split("__")[0] for cat in gt_categories]
                        gt_mid = [cat.split("__")[1] for cat in gt_categories]
                        gt_minor = [cat.split("__")[2] for cat in gt_categories]
                        
                        pred_major = [cat.split("__")[0] for cat in pred_categories]
                        pred_mid = [cat.split("__")[1] for cat in pred_categories]
                        pred_minor = [cat.split("__")[2] for cat in pred_categories]
                        
                        # Check for overlap
                        if set(gt_major) & set(pred_major):
                            major_category_correct += 1
                        if set(gt_mid) & set(pred_mid):
                            mid_category_correct += 1
                        if set(gt_minor) & set(pred_minor):
                            minor_category_correct += 1
                        
                        # Full category match
                        if set(gt_categories) & set(pred_categories):
                            category_correct += 1
            
            # Store result
            result = {
                "sample_id": idx,
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_valid_json": is_valid_json,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            results.append({
                "sample_id": idx,
                "error": str(e),
                "input": sample.get("input", ""),
            })
    
    # Calculate final metrics
    metrics = {
        "total_samples": total_samples,
        "total_claims": total_claims,
        "claim_accuracy": (claim_correct / total_samples * 100) if total_samples > 0 else 0,
        "category_accuracy": (category_correct / total_samples * 100) if total_samples > 0 else 0,
        "major_category_accuracy": (major_category_correct / total_claims * 100) if total_claims > 0 else 0,
        "mid_category_accuracy": (mid_category_correct / total_claims * 100) if total_claims > 0 else 0,
        "minor_category_accuracy": (minor_category_correct / total_claims * 100) if total_claims > 0 else 0,
    }
    
    # Save results
    output_data = {
        "metrics": metrics,
        "results": results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE METRICS")
    logger.info("=" * 80)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total claims: {total_claims}")
    logger.info(f"Claim accuracy: {metrics['claim_accuracy']:.2f}%")
    logger.info(f"Category accuracy: {metrics['category_accuracy']:.2f}%")
    logger.info(f"Major category accuracy: {metrics['major_category_accuracy']:.2f}%")
    logger.info(f"Mid category accuracy: {metrics['mid_category_accuracy']:.2f}%")
    logger.info(f"Minor category accuracy: {metrics['minor_category_accuracy']:.2f}%")
    logger.info("=" * 80)
    
    return results, metrics


def main():
    """Main inference function."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("BEST MODEL INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model_name}")
    logger.info(f"Adapter path: {args.model_dir}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"Max seq length: {args.max_seq_length}")
    logger.info(f"Load in 4-bit: {args.load_in_4bit}")
    
    # Load and display best model info if available
    best_model_info_path = os.path.join(os.path.dirname(args.model_dir), "best_model_info.json")
    if os.path.exists(best_model_info_path):
        try:
            with open(best_model_info_path, "r") as f:
                best_info = json.load(f)
            logger.info("-" * 80)
            logger.info("BEST MODEL CHECKPOINT INFO:")
            logger.info(f"  Claim Accuracy: {best_info.get('best_claim_accuracy', 'N/A'):.2f}%")
            logger.info(f"  Training Step: {best_info.get('best_model_step', 'N/A')}")
            logger.info(f"  Timestamp: {best_info.get('timestamp', 'N/A')}")
        except Exception as e:
            logger.warning(f"Could not load best model info: {e}")
    
    logger.info("=" * 80)
    
    # Load model
    model, tokenizer = load_model(
        base_model_name=args.base_model_name,
        adapter_path=args.model_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # Load test data
    dataset = load_test_data(args.test_data, args.max_samples)
    
    # Run inference
    results, metrics = run_inference(model, tokenizer, dataset, args.output_file)
    
    logger.info("\nInference complete! ✅")
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
