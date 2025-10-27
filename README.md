# CESCO sLLM Fine-tuning with SageMaker and LoRA

Fine-tune a small language model (sLLM) on AWS SageMaker using LoRA (Low-Rank Adaptation) for customer VOC classification.

## 🎯 Objective

Train a model that takes customer service input text and outputs structured JSON with:
- **categories**: List of classifications (format: `대분류__중분류__소분류`)
- **claim_status**: "claim" or "non-claim"
- **summary**: Brief Korean summary
- **bug_type**: Detected bug/pest type or null
- **keywords**: Up to 3 relevant keywords
- **evidences**: Supporting evidence for each category

## 🏗️ Architecture

- **Training Instance**: `ml.p4d.24xlarge` (8x A100 GPUs, 1.1TB RAM)
- **Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Base Model**: Meta Llama-2-7B or similar sLLM
- **Framework**: HuggingFace Transformers + PEFT + SageMaker

## 📁 Project Structure

```
cesco_sLLM/
├── main.py                          # Main orchestration script
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
├── assets/
│   ├── prompt_instructions.txt      # Instruction template
│   ├── voc_category_final.csv      # Available categories
│   └── bugs_df.csv                 # Bug/pest types
├── data/
│   ├── processed_train_dataset.json # Training data
│   ├── processed_test_dataset.json  # Test data
│   ├── train.jsonl                 # Prepared training data
│   └── test.jsonl                  # Prepared test data
└── src/
    ├── data_preparation.py         # Data preprocessing
    ├── training_script.py          # SageMaker training script
    ├── sagemaker_training.py       # Training job orchestration
    └── inference.py                # Inference utilities
```

## 🚀 Quick Start

### Prerequisites

1. **AWS Setup**
   ```bash
   # Configure AWS credentials
   aws configure
   
   # Create SageMaker execution role (if needed)
   # IAM role should have SageMakerFullAccess and S3 access
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables** (optional)
   ```bash
   export WANDB_API_KEY="your-wandb-key"  # For experiment tracking
   ```

### Step 1: Prepare Training Data

Convert processed datasets to instruction fine-tuning format:

```bash
python main.py --prepare-data
```

This creates `data/train.jsonl` and `data/test.jsonl` with instruction-response pairs.

### Step 2: Launch Training on SageMaker

```bash
python main.py --train \
  --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
  --bucket your-s3-bucket-name \
  --model-name meta-llama/Llama-2-7b-hf \
  --instance-type ml.p4d.24xlarge \
  --epochs 3 \
  --batch-size 4 \
  --lora-r 16 \
  --lora-alpha 32
```

**Key Parameters:**
- `--role`: SageMaker execution role ARN
- `--bucket`: S3 bucket for data and model artifacts
- `--instance-type`: Training instance (default: ml.p4d.24xlarge)
- `--epochs`: Number of training epochs
- `--batch-size`: Per-device batch size
- `--lora-r`: LoRA rank (lower = fewer parameters)
- `--lora-alpha`: LoRA scaling parameter
- `--use-spot`: Use spot instances for cost savings
- `--deploy`: Automatically deploy after training

### Step 3: Run Everything at Once

```bash
python main.py --all \
  --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
  --bucket your-s3-bucket-name \
  --deploy
```

## 💰 Cost Optimization

### Using Spot Instances

Save up to 70% on training costs:

```bash
python main.py --train \
  --role <role-arn> \
  --bucket <bucket> \
  --use-spot
```

### Instance Type Comparison

| Instance Type | GPUs | GPU Memory | Hourly Cost* | Use Case |
|--------------|------|------------|--------------|----------|
| ml.p4d.24xlarge | 8x A100 | 320GB | ~$32 | Large models, fast training |
| ml.g5.12xlarge | 4x A10G | 96GB | ~$5 | Medium models |
| ml.g5.2xlarge | 1x A10G | 24GB | ~$1.5 | Inference, small models |

*Approximate on-demand pricing in us-east-1

## 🔧 Advanced Configuration

### Custom Hyperparameters

Create `configs.txt` or pass directly:

```bash
python main.py --train \
  --role <role> \
  --bucket <bucket> \
  --epochs 5 \
  --batch-size 2 \
  --learning-rate 3e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --instance-count 2  # Multi-node training
```

### LoRA Configuration

Adjust these parameters based on model size and task complexity:

- **lora_r** (rank): 8-64 (default: 16)
  - Lower = fewer parameters, faster training
  - Higher = more capacity, better performance
  
- **lora_alpha**: Usually 2× lora_r (default: 32)
  - Controls scaling of LoRA weights
  
- **target_modules**: Which layers to adapt
  - Default: `q_proj,v_proj,k_proj,o_proj` (attention layers)
  - Can include: `gate_proj,up_proj,down_proj` (MLP layers)

### Training Monitoring

Training metrics are logged to:
- **CloudWatch Logs**: Real-time training logs
- **Weights & Biases**: Experiment tracking (if configured)
- **S3**: Model checkpoints and final artifacts

```bash
# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# Or use W&B dashboard
# https://wandb.ai/your-project/cesco-sllm-finetuning
```

## 📊 Model Inference

### Local Inference

```python
from src.inference import LocalInference

# Initialize model
inference = LocalInference(
    base_model_name="meta-llama/Llama-2-7b-hf",
    lora_weights_path="path/to/lora/weights"
)

# Predict
result = inference.predict("정수기에서 물이 안 나옵니다.")
print(result)
# {
#   "claim_status": "claim",
#   "summary": "정수기 출수 불가 문제",
#   "bug_type": null,
#   "keywords": ["정수기", "출수", "고장"],
#   "categories": ["제품__정수기__출수 불가"],
#   "evidences": ["물이 안 나옴"]
# }
```

### SageMaker Endpoint Inference

```python
from src.inference import SageMakerInference

# Connect to endpoint
inference = SageMakerInference(endpoint_name="cesco-sllm-endpoint-20241026")

# Predict
result = inference.predict("바퀴벌레가 나왔어요")
print(result)
```

### Deploy Model

```bash
# During training
python main.py --train --role <role> --bucket <bucket> --deploy

# Or deploy separately
python -c "
from src.sagemaker_training import SageMakerTrainer
trainer = SageMakerTrainer()
# Use estimator from training job
predictor = trainer.deploy_model(
    estimator,
    instance_type='ml.g5.2xlarge'
)
print(f'Endpoint: {predictor.endpoint_name}')
"
```

## 🧪 Testing

Test the pipeline end-to-end:

```python
from src.data_preparation import prepare_training_data
from src.inference import test_inference

# Test data preparation
prepare_training_data(
    input_file="data/processed_test_dataset.json",
    output_file="data/test_sample.jsonl",
    format_type="jsonl"
)

# Test inference (requires trained model)
test_inference()
```

## 📈 Performance Metrics

The model is evaluated on:
- **Accuracy**: Category prediction accuracy
- **F1 Score**: Per-category F1 scores
- **ROUGE**: Summary quality
- **Exact Match**: Full JSON exact match rate

Metrics are saved to `s3://<bucket>/cesco-sllm/output/metrics.json`

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use smaller `lora_r`

2. **Slow Training**
   - Increase batch size if memory allows
   - Use multiple instances (`--instance-count 2`)
   - Enable FP16 (enabled by default)

3. **Poor Performance**
   - Increase `lora_r` and `lora_alpha`
   - Train for more epochs
   - Adjust learning rate (try 1e-4 to 5e-4)

4. **Cost Too High**
   - Use `--use-spot` for spot instances
   - Use smaller instance type for initial experiments
   - Set `keep_alive_period_in_seconds=0` to avoid warm pool costs

## 📚 Additional Resources

- [SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 📝 License

This project is proprietary to CESCO.

## 🤝 Contributing

For questions or issues, contact the ML team.

---

**Last Updated**: 2025-10-26
