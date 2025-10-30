#!/usr/bin/env python3
"""
Quick test script to run inference on 10 samples with enhanced W&B logging.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the inference script with test parameters."""
    
    # Path to the inference script
    script_path = Path(__file__).parent / "src" / "inference_merged_model.py"
    
    # Test parameters
    cmd = [
        sys.executable, str(script_path),
        "--model_path", "./outputs/merged_model",
        "--test_data_path", "./data/unsloth_test_dataset.json", 
        "--output_csv", "./outputs/test_inference_results.csv",
        "--max_samples", "10",
        "--use_wandb", "True",
        "--wandb_project", "cesco-sllm-test",
        "--wandb_run_name", "10-sample-test",
        "--log_samples_to_wandb", "0",  # Log all samples
        "--log_sample_text", "True",
        "--max_new_tokens", "512",
        "--temperature", "0.7",
        "--do_sample", "True"
    ]
    
    print("Running inference test with 10 samples...")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nInference completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nInference failed with error code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Could not find inference script at {script_path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
