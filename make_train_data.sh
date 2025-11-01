#!/bin/bash
set -e  # Exit on error

echo "============================================"
echo "Checking system and clearing memory..."
echo "============================================"

# Clear HuggingFace cache locks
rm -rf ~/.cache/huggingface/hub/.locks 2>/dev/null || true
rm -rf ~/.cache/huggingface/hub/*lock* 2>/dev/null || true

# Kill any zombie Python processes
echo "Cleaning up any previous Python processes..."
pkill -9 -f "inference_best_model.py" 2>/dev/null || true
pkill -9 -f "unsloth_training.py" 2>/dev/null || true
pkill -9 -f "uv run python" 2>/dev/null || true
sleep 3

# Force Python garbage collection
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Double check they're gone
echo "Verifying cleanup..."
if ps aux | grep -E "inference|unsloth" | grep -v grep > /dev/null; then
    echo "Warning: Some processes still running. Forcing cleanup..."
    killall -9 python3 2>/dev/null || true
    sleep 2
fi

echo "Step 1: Running inference..."
uv run python inference_best_model.py --max_samples 10
echo "Inference completed. Waiting for cleanup..."
sleep 2

echo "Step 2: Creating modified training dataset..."
uv run python make_train_dataset.py
echo "Dataset created. Waiting for cleanup..."
sleep 2

# Clear Python cache and force cleanup
echo "Step 3: Clearing cache and memory..."
rm -rf __pycache__ src/__pycache__ 2>/dev/null || true
sync
sleep 3

echo "Step 4: Starting training..."
uv run python src/unsloth_training.py --train_dir data/modified_unsloth_train.json --max_steps 10