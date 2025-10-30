#!/bin/bash

# Inference script for Unsloth models
# Usage: ./run_inference.sh [mode] [options...]
# Modes: 
#   merged - Load merged model (default)
#   lora   - Load LoRA adapters

MODE=${1:-"merged"}
shift  # Remove first argument

case $MODE in
    "merged")
        echo "=== MERGED MODEL INFERENCE ==="
        MODEL_PATH=${1:-"./outputs/merged_model"}
        TEST_DATA_PATH=${2:-"./data/unsloth_test_dataset.json"}
        OUTPUT_CSV=${3:-"./outputs/inference_results.csv"}
        
        echo "Running merged model inference with:"
        echo "  Model path: $MODEL_PATH"
        echo "  Test data: $TEST_DATA_PATH"
        echo "  Output CSV: $OUTPUT_CSV"
        echo ""
        
        # Create output directory if it doesn't exist
        mkdir -p $(dirname "$OUTPUT_CSV")
        
        python src/inference_merged_model.py \
            --model_path "$MODEL_PATH" \
            --test_data_path "$TEST_DATA_PATH" \
            --output_csv "$OUTPUT_CSV" \
            --max_new_tokens 512 \
            --temperature 0.7 \
            --top_p 0.9 \
            --do_sample true \
            --use_unsloth true
        ;;
        
    "lora")
        echo "=== LORA ADAPTER INFERENCE ==="
        BASE_MODEL=${1:-"unsloth/Meta-Llama-3.1-8B"}
        ADAPTER_PATH=${2:-"./outputs"}
        TEST_DATA_PATH=${3:-"./data/unsloth_test_dataset.json"}
        OUTPUT_CSV=${4:-"./outputs/inference_results_lora.csv"}
        
        echo "Running LoRA adapter inference with:"
        echo "  Base model: $BASE_MODEL"
        echo "  Adapter path: $ADAPTER_PATH"
        echo "  Test data: $TEST_DATA_PATH"
        echo "  Output CSV: $OUTPUT_CSV"
        echo ""
        
        # Create output directory if it doesn't exist
        mkdir -p $(dirname "$OUTPUT_CSV")
        
        python src/inference_lora_adapters.py \
            --base_model "$BASE_MODEL" \
            --adapter_path "$ADAPTER_PATH" \
            --test_data_path "$TEST_DATA_PATH" \
            --output_csv "$OUTPUT_CSV" \
            --max_new_tokens 512 \
            --temperature 0.7 \
            --top_p 0.9 \
            --do_sample true
        ;;
        
    *)
        echo "Usage: $0 [merged|lora] [options...]"
        echo ""
        echo "Merged model mode:"
        echo "  $0 merged [model_path] [test_data_path] [output_csv]"
        echo "  Default: $0 merged ./outputs/merged_model ./data/unsloth_test_dataset.json ./outputs/inference_results.csv"
        echo ""
        echo "LoRA adapter mode:"
        echo "  $0 lora [base_model] [adapter_path] [test_data_path] [output_csv]"
        echo "  Default: $0 lora unsloth/Meta-Llama-3.1-8B ./outputs ./data/unsloth_test_dataset.json ./outputs/inference_results_lora.csv"
        exit 1
        ;;
esac
