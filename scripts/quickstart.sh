#!/bin/bash

# Quick start script for CESCO sLLM fine-tuning on SageMaker
# Usage: ./scripts/quickstart.sh

set -e

echo "======================================"
echo "CESCO sLLM Fine-tuning Quick Start"
echo "======================================"
echo ""

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials configured"

# Check if required files exist
if [ ! -f "data/processed_train_dataset.json" ]; then
    echo "❌ Training data not found: data/processed_train_dataset.json"
    exit 1
fi

if [ ! -f "data/processed_test_dataset.json" ]; then
    echo "❌ Test data not found: data/processed_test_dataset.json"
    exit 1
fi

echo "✅ Data files found"

# Check for required environment variables
if [ -z "$SAGEMAKER_ROLE_ARN" ]; then
    echo "⚠️  SAGEMAKER_ROLE_ARN not set. Please provide it:"
    read -p "Enter SageMaker execution role ARN: " SAGEMAKER_ROLE_ARN
    export SAGEMAKER_ROLE_ARN
fi

if [ -z "$S3_BUCKET" ]; then
    echo "⚠️  S3_BUCKET not set. Using default bucket from SageMaker session."
    S3_BUCKET="default"
fi

echo ""
echo "Configuration:"
echo "  Role ARN: $SAGEMAKER_ROLE_ARN"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Instance Type: ml.p4d.24xlarge"
echo "  Method: LoRA"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing training data..."
python main.py --prepare-data

# Step 2: Launch training
echo ""
echo "Step 2: Launching SageMaker training job..."

ARGS="--train --role $SAGEMAKER_ROLE_ARN"

if [ "$S3_BUCKET" != "default" ]; then
    ARGS="$ARGS --bucket $S3_BUCKET"
fi

# Check for spot instances option
if [ "$USE_SPOT" = "true" ]; then
    echo "  Using spot instances for cost savings"
    ARGS="$ARGS --use-spot"
fi

# Execute training
python main.py $ARGS

echo ""
echo "======================================"
echo "✅ Training job submitted successfully!"
echo "======================================"
echo ""
echo "Monitor your training job:"
echo "  - AWS Console: https://console.aws.amazon.com/sagemaker/home#/jobs"
echo "  - CloudWatch Logs: aws logs tail /aws/sagemaker/TrainingJobs --follow"
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "  - Weights & Biases: https://wandb.ai"
fi
echo ""
echo "To deploy the model after training:"
echo "  python main.py --train --role $SAGEMAKER_ROLE_ARN --deploy"
echo ""
