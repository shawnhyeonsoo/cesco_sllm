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
        "--model_dir",
        type=str,
        default="./outputs/best_model/merged_model",
        help="Path to the best merged model directory"
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
    return parser.parse_args()


def load_model(model_path: str):
    """Load the best saved model using Unsloth."""
    try:
        from unsloth import FastLanguageModel
        
        logger.info(f"Loading model from {model_path}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Expected structure:\n"
                f"  {model_path}/\n"
                f"    ├── config.json\n"
                f"    ├── model-*.safetensors\n"
                f"    └── tokenizer files..."
            )
        
        # Load the merged model (no need for LoRA adapters)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=8096,
            dtype=None,
            load_in_4bit=True,  # Use 4-bit for faster inference
        )
        
        # Set model to inference mode
        FastLanguageModel.for_inference(model)
        
        logger.info("Model loaded successfully!")
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
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
    logger.info(f"Model path: {args.model_dir}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info("=" * 80)
    
    # Load model
    model, tokenizer = load_model(args.model_dir)
    
    # Load test data
    dataset = load_test_data(args.test_data, args.max_samples)
    
    # Run inference
    results, metrics = run_inference(model, tokenizer, dataset, args.output_file)
    
    logger.info("\nInference complete! ✅")
    logger.info(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
