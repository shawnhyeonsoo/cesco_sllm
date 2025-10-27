#!/bin/bash

# Example training commands for CESCO sLLM fine-tuning

echo "CESCO sLLM Training Examples"
echo "============================"
echo ""

# 1. Show current configuration
echo "1. Display configuration:"
echo "   python main.py --show-config"
echo ""

# 2. Prepare data only
echo "2. Prepare data only:"
echo "   python main.py --prepare-data"
echo ""

# 3. Basic training (uses all defaults from config.py)
echo "3. Basic training with defaults:"
echo "   python main.py --train"
echo ""

# 4. Full pipeline (prepare data + train)
echo "4. Run full pipeline:"
echo "   python main.py --all"
echo ""

# 5. Training with custom hyperparameters
echo "5. Custom hyperparameters:"
echo "   python main.py --train --epochs 5 --batch-size 2 --learning-rate 3e-4"
echo ""

# 6. Cost-optimized training with spot instances
echo "6. Use spot instances (70% cost savings):"
echo "   python main.py --train --use-spot"
echo ""

# 7. Training with deployment
echo "7. Train and deploy:"
echo "   python main.py --all --deploy"
echo ""

# 8. Experiment with different LoRA configurations
echo "8. Experiment with LoRA settings:"
echo "   python main.py --train --lora-r 32 --lora-alpha 64"
echo ""

# 9. Quick test run (1 epoch, spot instance)
echo "9. Quick test run:"
echo "   python main.py --train --epochs 1 --use-spot"
echo ""

# 10. Full production training
echo "10. Production training:"
echo "    python main.py --all --epochs 5 --batch-size 4 --deploy"
echo ""

echo "============================"
echo "Your AWS Configuration:"
echo "  Role: arn:aws:iam::326614947732:role/AmazonSageMakerFullAccess"
echo "  Bucket: amazon-sagemaker-326614947732-us-east-1-b6aed9d1f258"
echo "  Instance: ml.p4d.24xlarge"
echo ""
echo "Ready to start? Run:"
echo "  python main.py --all"
