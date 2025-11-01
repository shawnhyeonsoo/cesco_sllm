#!/bin/bash
set -e  # Exit on error

echo "Step 1: Running inference..."
uv run python inference_best_model.py --max_samples 10

echo "Step 2: Creating modified training dataset..."
uv run python make_train_dataset.py

# Force garbage collection and clear any cached memory
echo "Step 3: Clearing memory before training..."
sleep 2

echo "Step 4: Starting training..."
uv run python src/unsloth_training.py --train_data_file data/modified_unsloth_train.json