uv run python inference_best_model.py --max_samples 10
uv run python make_train_dataset.py
uv run python src/unsloth_training.py --train_data_file data/modified_unsloth_train.json