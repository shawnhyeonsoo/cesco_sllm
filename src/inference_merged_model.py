from unsloth import FastLanguageModel
import json
import argparse
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


UNSLOTH_AVAILABLE = True
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/merged_model",
        help="Path to the merged model directory",
    )
    parser.add_argument(
        "--use_unsloth",
        type=bool,
        default=True,
        help="Whether to use Unsloth for model loading (recommended)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/unsloth_test_dataset.json",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./outputs/inference_results.csv",
        help="Path to save CSV results",
    )

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", type=bool, default=True)

    # Processing arguments
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process",
    )

    # W&B arguments
    parser.add_argument(
        "--use_wandb", type=bool, default=True, help="Whether to log to W&B"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cesco-sllm-inference",
        help="W&B project name",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--log_samples_to_wandb",
        type=int,
        default=0,
        help="Number of samples to log to W&B (0 for all)",
    )
    parser.add_argument(
        "--log_sample_text",
        type=bool,
        default=True,
        help="Whether to log sample text to W&B",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_path, use_unsloth=True):
    """Load the merged model and tokenizer."""
    logger.info(f"Loading model from {model_path}")

    # Check if model path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Try loading with Unsloth first if available (for better compatibility)
    if UNSLOTH_AVAILABLE and use_unsloth:
        try:
            logger.info("Attempting to load with Unsloth FastLanguageModel...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=8096,
                dtype=None,  # Auto-detect
                load_in_4bit=False,  # Merged model should be full precision
            )
            # Set model to inference mode
            FastLanguageModel.for_inference(model)
            logger.info("Model loaded successfully with Unsloth!")
        except Exception as e:
            logger.warning(f"Failed to load with Unsloth: {e}")
            logger.info("Falling back to standard transformers loading...")
            use_unsloth = False

    # Fallback to standard transformers loading
    if not (UNSLOTH_AVAILABLE and use_unsloth):
        logger.info("Loading with standard transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully with transformers!")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_test_data(test_path):
    """Load test dataset."""
    logger.info(f"Loading test data from {test_path}")

    if test_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files={"test": test_path}, split="test")
    else:
        dataset = load_dataset("json", data_files=test_path, split="train")

    logger.info(f"Loaded {len(dataset)} test examples")
    return dataset


def create_prompt(instruction, input_text):
    """Create Alpaca-style prompt for inference."""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    return alpaca_prompt.format(instruction, input_text)


def extract_json_from_response(response_text):
    """Extract JSON from model response."""
    try:
        # Find JSON in the response
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            return parsed, json_str
        else:
            return None, response_text
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None, response_text


def run_inference(model, tokenizer, dataset, args):
    """Run inference on the test dataset."""
    logger.info("Starting inference...")

    results = []
    max_samples = args.max_samples or len(dataset)
    wandb_samples = []  # Store samples for W&B logging

    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Running inference")):
            if i >= max_samples:
                break

            # Create prompt
            instruction = sample["instruction"]
            input_text = sample["input"]
            ground_truth = sample["output"]

            prompt = create_prompt(instruction, input_text)

            # Tokenize
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            ).to(model.device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the response part
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text[len(prompt) :].strip()

            # Parse JSON from response
            parsed_json, raw_response = extract_json_from_response(response)

            # Parse ground truth JSON
            try:
                gt_json = json.loads(ground_truth)
            except json.JSONDecodeError:
                gt_json = None

            # Store results
            result = {
                "sample_id": i,
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "generated_response": raw_response,
                "parsed_prediction": json.dumps(parsed_json, ensure_ascii=False)
                if parsed_json
                else None,
                "ground_truth_parsed": json.dumps(gt_json, ensure_ascii=False)
                if gt_json
                else None,
            }

            # Add individual fields if parsing was successful
            if parsed_json:
                result.update(
                    {
                        "pred_claim_status": parsed_json.get("claim_status"),
                        "pred_summary": parsed_json.get("summary"),
                        "pred_bug_type": parsed_json.get("bug_type"),
                        "pred_keywords": json.dumps(
                            parsed_json.get("keywords", []), ensure_ascii=False
                        ),
                        "pred_categories": json.dumps(
                            parsed_json.get("categories", []), ensure_ascii=False
                        ),
                        "pred_evidences": json.dumps(
                            parsed_json.get("evidences", []), ensure_ascii=False
                        ),
                    }
                )

            if gt_json:
                result.update(
                    {
                        "gt_claim_status": gt_json.get("claim_status"),
                        "gt_summary": gt_json.get("summary"),
                        "gt_bug_type": gt_json.get("bug_type"),
                        "gt_keywords": json.dumps(
                            gt_json.get("keywords", []), ensure_ascii=False
                        ),
                        "gt_categories": json.dumps(
                            gt_json.get("categories", []), ensure_ascii=False
                        ),
                        "gt_evidences": json.dumps(
                            gt_json.get("evidences", []), ensure_ascii=False
                        ),
                    }
                )

            results.append(result)

            # Enhanced W&B logging with detailed step-by-step information
            if args.use_wandb:
                try:
                    import wandb

                    # Log individual sample metrics
                    claim_match = None
                    if result.get("pred_claim_status") and result.get(
                        "gt_claim_status"
                    ):
                        claim_match = result.get("pred_claim_status") == result.get(
                            "gt_claim_status"
                        )

                    # Enhanced logging with full model output details
                    sample_log = {
                        # Step identification
                        "sample_id": i,
                        "step": i + 1,
                        
                        # Input information
                        "instruction": instruction,
                        "input_text": input_text,
                        "prompt_length": len(prompt),
                        
                        # Model output
                        "raw_generated_text": generated_text,
                        "extracted_response": raw_response,
                        "response_length": len(raw_response),
                        
                        # JSON parsing results
                        "valid_json": 1.0 if parsed_json is not None else 0.0,
                        "parsed_prediction": json.dumps(parsed_json, ensure_ascii=False) if parsed_json else "null",
                        
                        # Ground truth
                        "ground_truth": ground_truth,
                        "ground_truth_parsed": json.dumps(gt_json, ensure_ascii=False) if gt_json else "null",
                        
                        # Predictions vs Ground Truth
                        "pred_claim_status": result.get("pred_claim_status") or "unknown",
                        "gt_claim_status": result.get("gt_claim_status") or "unknown",
                        "claim_match": 1.0 if claim_match else 0.0 if claim_match is False else 0.5,
                        
                        # Additional fields
                        "pred_summary": result.get("pred_summary") or "",
                        "gt_summary": result.get("gt_summary") or "",
                        "pred_bug_type": result.get("pred_bug_type") or "",
                        "gt_bug_type": result.get("gt_bug_type") or "",
                        "pred_keywords": result.get("pred_keywords") or "[]",
                        "gt_keywords": result.get("gt_keywords") or "[]",
                        "pred_categories": result.get("pred_categories") or "[]",
                        "gt_categories": result.get("gt_categories") or "[]",
                        "pred_evidences": result.get("pred_evidences") or "[]",
                        "gt_evidences": result.get("gt_evidences") or "[]",
                    }

                    # Add category match if available
                    if result.get("pred_categories") and result.get("gt_categories"):
                        try:
                            pred_cats = json.loads(result["pred_categories"])
                            gt_cats = json.loads(result["gt_categories"])
                            category_match = set(gt_cats).issubset(set(pred_cats))
                            sample_log["category_match"] = 1.0 if category_match else 0.0
                        except Exception:
                            sample_log["category_match"] = 0.0
                    else:
                        sample_log["category_match"] = 0.0

                    # Log with step explicitly set
                    wandb.log(sample_log, step=i)

                    # Detailed logging for every sample
                    logger.info("=" * 60)
                    logger.info(f"SAMPLE {i + 1} - W&B LOGGED")
                    logger.info("=" * 60)
                    logger.info(f"Input: {input_text[:200]}...")
                    logger.info(f"Generated Response: {raw_response[:200]}...")
                    logger.info(f"Valid JSON: {parsed_json is not None}")
                    logger.info(f"Claim Match: {claim_match}")
                    logger.info(f"Predicted Claim: {result.get('pred_claim_status')}")
                    logger.info(f"Ground Truth Claim: {result.get('gt_claim_status')}")
                    logger.info("=" * 60)

                    # Store for final table logging
                    if (
                        args.log_samples_to_wandb == 0
                        or len(wandb_samples) < args.log_samples_to_wandb
                    ):
                        wandb_sample = {
                            "sample_id": i,
                            "input_text": input_text,
                            "generated_response": raw_response,
                            "ground_truth": ground_truth,
                            "claim_match": claim_match,
                            "valid_json": parsed_json is not None,
                            "pred_claim_status": result.get("pred_claim_status"),
                            "gt_claim_status": result.get("gt_claim_status"),
                        }
                        wandb_samples.append(wandb_sample)

                except Exception as e:
                    logger.warning(f"Failed to log sample {i} to W&B: {e}")

            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{max_samples} samples")

    # Log comprehensive samples table to W&B
    if args.use_wandb and wandb_samples:
        try:
            import wandb

            logger.info(f"Logging {len(wandb_samples)} samples to W&B...")

            # Create a comprehensive W&B table with input-output pairs
            columns = [
                "sample_id",
                "input_text",
                "generated_response", 
                "ground_truth",
                "pred_claim_status",
                "gt_claim_status",
                "claim_match",
                "valid_json",
            ]
            table_data = []

            for sample in wandb_samples:
                table_data.append(
                    [
                        sample["sample_id"],
                        sample["input_text"],
                        sample["generated_response"],
                        sample["ground_truth"],
                        sample.get("pred_claim_status", "unknown"),
                        sample.get("gt_claim_status", "unknown"),
                        sample["claim_match"],
                        sample["valid_json"],
                    ]
                )

            table = wandb.Table(columns=columns, data=table_data)
            wandb.log({"final_inference_results": table})
            
            # Also save a detailed input-output table
            input_output_columns = ["sample_id", "full_input", "full_output"]
            input_output_data = []
            
            for i, sample in enumerate(wandb_samples):
                input_output_data.append([
                    sample["sample_id"],
                    sample["input_text"],
                    sample["generated_response"]
                ])
            
            input_output_table = wandb.Table(columns=input_output_columns, data=input_output_data)
            wandb.log({"input_output_pairs": input_output_table})
            
            logger.info("Comprehensive samples logged to W&B successfully!")

        except Exception as e:
            logger.warning(f"Failed to log samples to W&B: {e}")

    return results


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    metrics = {}

    total_samples = len(results)
    claim_correct = 0
    category_matches = 0
    valid_predictions = 0

    for result in results:
        # Count valid predictions (successfully parsed JSON)
        if result.get("parsed_prediction"):
            valid_predictions += 1

        # Claim status accuracy
        if (
            result.get("pred_claim_status")
            and result.get("gt_claim_status")
            and result["pred_claim_status"] == result["gt_claim_status"]
        ):
            claim_correct += 1

        # Category accuracy (if ground truth categories are subset of predicted)
        try:
            if result.get("pred_categories") and result.get("gt_categories"):
                pred_cats = json.loads(result["pred_categories"])
                gt_cats = json.loads(result["gt_categories"])
                if set(gt_cats).issubset(set(pred_cats)):
                    category_matches += 1
        except Exception:
            pass

    metrics = {
        "total_samples": total_samples,
        "valid_predictions": valid_predictions,
        "valid_prediction_rate": (valid_predictions / total_samples * 100)
        if total_samples > 0
        else 0,
        "claim_accuracy": (claim_correct / total_samples * 100)
        if total_samples > 0
        else 0,
        "category_accuracy": (category_matches / total_samples * 100)
        if total_samples > 0
        else 0,
    }

    return metrics


def main():
    """Main inference function."""
    args = parse_args()

    # Initialize W&B if requested
    if args.use_wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or f"inference-{Path(args.model_path).name}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model_path": args.model_path,
                    "test_data_path": args.test_data_path,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "do_sample": args.do_sample,
                    "max_samples": args.max_samples,
                    "use_unsloth": args.use_unsloth,
                },
            )
            logger.info(
                f"W&B initialized: project={args.wandb_project}, run={run_name}"
            )
            logger.info(f"W&B run URL: {wandb.run.get_url()}")
        except ImportError:
            logger.warning(
                "W&B requested but not installed. Continuing without W&B logging."
            )
            args.use_wandb = False

    # Create output directory
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("INFERENCE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data: {args.test_data_path}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Use W&B: {args.use_wandb}")
    if args.use_wandb:
        logger.info(f"Log samples to W&B: {args.log_samples_to_wandb}")
    logger.info("=" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.use_unsloth)

    # Load test data
    test_dataset = load_test_data(args.test_data_path)

    # Run inference
    results = run_inference(model, tokenizer, test_dataset, args)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Log metrics
    logger.info("=" * 80)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {metrics['total_samples']}")
    logger.info(
        f"Valid predictions: {metrics['valid_predictions']} ({metrics['valid_prediction_rate']:.2f}%)"
    )
    logger.info(f"Claim accuracy: {metrics['claim_accuracy']:.2f}%")
    logger.info(f"Category accuracy: {metrics['category_accuracy']:.2f}%")
    logger.info("=" * 80)

    # Log metrics to W&B
    if args.use_wandb:
        try:
            import wandb

            wandb.log(
                {
                    "total_samples": metrics["total_samples"],
                    "valid_predictions": metrics["valid_predictions"],
                    "valid_prediction_rate": metrics["valid_prediction_rate"],
                    "claim_accuracy": metrics["claim_accuracy"],
                    "category_accuracy": metrics["category_accuracy"],
                }
            )
            logger.info("Metrics logged to W&B!")
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    logger.info(f"Results saved to {args.output_csv}")

    # Save metrics to JSON
    metrics_path = args.output_csv.replace(".csv", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save detailed input-output pairs in readable format
    input_output_path = args.output_csv.replace(".csv", "_input_output_pairs.json")
    input_output_pairs = []
    
    for i, result in enumerate(results):
        pair = {
            "sample_id": i,
            "input": {
                "instruction": result["instruction"],
                "input_text": result["input"]
            },
            "output": {
                "generated_response": result["generated_response"],
                "parsed_prediction": result.get("parsed_prediction"),
                "valid_json": result.get("parsed_prediction") is not None
            },
            "ground_truth": {
                "expected_output": result["ground_truth"],
                "parsed_ground_truth": result.get("ground_truth_parsed")
            },
            "evaluation": {
                "claim_match": result.get("pred_claim_status") == result.get("gt_claim_status") 
                            if result.get("pred_claim_status") and result.get("gt_claim_status") else None,
                "pred_claim_status": result.get("pred_claim_status"),
                "gt_claim_status": result.get("gt_claim_status")
            }
        }
        input_output_pairs.append(pair)
    
    with open(input_output_path, "w", encoding="utf-8") as f:
        json.dump(input_output_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Input-output pairs saved to {input_output_path}")
    
    # Print final summary
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY - INPUT/OUTPUT PAIRS")  
    logger.info("=" * 80)
    for i, pair in enumerate(input_output_pairs):
        logger.info(f"\nSAMPLE {i + 1}:")
        logger.info(f"INPUT: {pair['input']['input_text'][:150]}...")
        logger.info(f"OUTPUT: {pair['output']['generated_response'][:150]}...")
        logger.info(f"VALID JSON: {pair['output']['valid_json']}")
        logger.info(f"CLAIM MATCH: {pair['evaluation']['claim_match']}")
    logger.info("=" * 80)

    # Finish W&B run
    if args.use_wandb:
        try:
            import wandb

            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
