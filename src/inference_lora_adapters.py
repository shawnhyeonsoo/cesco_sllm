"""
Alternative inference script that loads LoRA adapters directly.
Use this if the merged model has issues.
"""
import json
import argparse
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import torch
from datasets import load_dataset

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Error: Unsloth is required for LoRA adapter loading!")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B",
                       help="Base model name (same as used in training)")
    parser.add_argument("--adapter_path", type=str, default="./outputs", 
                       help="Path to LoRA adapter directory")
    parser.add_argument("--max_seq_length", type=int, default=8096)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    
    # Data arguments
    parser.add_argument("--test_data_path", type=str, default="./data/unsloth_test_dataset.json",
                       help="Path to test dataset")
    parser.add_argument("--output_csv", type=str, default="./outputs/inference_results_lora.csv",
                       help="Path to save CSV results")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", type=bool, default=True)
    
    # Processing arguments
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    
    return parser.parse_args()


def load_model_with_lora(base_model, adapter_path, max_seq_length, load_in_4bit):
    """Load base model and LoRA adapters."""
    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    
    # Check if adapter path exists
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    # Load base model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    # Load LoRA adapters
    logger.info("Loading LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Should match training config
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    
    # Load the trained weights
    try:
        model.load_adapter(adapter_path)
        logger.info("LoRA adapters loaded successfully!")
    except Exception as e:
        logger.warning(f"Failed to load adapters with load_adapter: {e}")
        # Try alternative loading method
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("LoRA adapters loaded with PeftModel!")
        except Exception as e2:
            logger.error(f"Failed to load adapters: {e2}")
            raise
    
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def load_test_data(test_path):
    """Load test dataset."""
    logger.info(f"Loading test data from {test_path}")
    
    if test_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files={'test': test_path}, split='test')
    else:
        dataset = load_dataset('json', data_files=test_path, split='train')
    
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


def run_inference(model, tokenizer, dataset, args):
    """Run inference on the test dataset."""
    logger.info("Starting inference...")
    
    results = []
    max_samples = args.max_samples or len(dataset)
    
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
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
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
                use_cache=True
            )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            # Store results
            result = {
                "sample_id": i,
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "generated_response": response,
            }
            
            results.append(result)
            
            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{max_samples} samples")
    
    return results


def main():
    """Main inference function."""
    args = parse_args()
    
    if not UNSLOTH_AVAILABLE:
        logger.error("Unsloth is required for LoRA adapter inference!")
        return
    
    # Create output directory
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("LORA ADAPTER INFERENCE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter path: {args.adapter_path}")
    logger.info(f"Test data: {args.test_data_path}")
    logger.info(f"Output CSV: {args.output_csv}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info("=" * 80)
    
    # Load model with LoRA adapters
    model, tokenizer = load_model_with_lora(
        args.base_model, 
        args.adapter_path, 
        args.max_seq_length, 
        args.load_in_4bit
    )
    
    # Load test data
    test_dataset = load_test_data(args.test_data_path)
    
    # Run inference
    results = run_inference(model, tokenizer, test_dataset, args)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False, encoding='utf-8')
    logger.info(f"Results saved to {args.output_csv}")
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
