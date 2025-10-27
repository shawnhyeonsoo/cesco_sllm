"""
Main orchestration script for CESCO sLLM fine-tuning pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preparation import prepare_training_data
from sagemaker_training import SageMakerTrainer
from config import Config


def prepare_data():
    """Prepare training and test data."""
    print("=" * 50)
    print("Step 1: Preparing training data")
    print("=" * 50)
    
    prepare_training_data(
        input_file="data/processed_train_dataset.json",
        output_file="data/train.jsonl",
        format_type="jsonl"
    )
    
    prepare_training_data(
        input_file="data/processed_test_dataset.json",
        output_file="data/test.jsonl",
        format_type="jsonl"
    )
    
    print("\n✅ Data preparation completed!")


def launch_training(args):
    """Launch SageMaker training job."""
    print("\n" + "=" * 50)
    print("Step 2: Launching SageMaker training job")
    print("=" * 50)
    
    # Use config defaults if not provided
    role = args.role or Config.SAGEMAKER_ROLE_ARN
    bucket = args.bucket or Config.S3_BUCKET
    region = args.region or Config.AWS_REGION
    
    # Initialize trainer
    trainer = SageMakerTrainer(
        role_arn=role,
        s3_bucket=bucket,
        region=region
    )
    
    # Upload data to S3
    print("\nUploading data to S3...")
    train_s3_uri, test_s3_uri = trainer.upload_data_to_s3(
        train_data_path="data/train.jsonl",
        test_data_path="data/test.jsonl"
    )
    
    # Configure hyperparameters (use Config defaults, override with args if provided)
    hyperparameters = Config.get_hyperparameters()
    
    # Override with command-line arguments if provided
    if args.epochs:
        hyperparameters["num_train_epochs"] = args.epochs
    if args.batch_size:
        hyperparameters["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        hyperparameters["learning_rate"] = args.learning_rate
    if args.lora_r:
        hyperparameters["lora_r"] = args.lora_r
    if args.lora_alpha:
        hyperparameters["lora_alpha"] = args.lora_alpha
    
    # Create training job
    print("\nCreating training job...")
    estimator = trainer.create_training_job(
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        model_name=args.model_name or Config.BASE_MODEL_NAME,
        instance_type=args.instance_type or Config.TRAINING_INSTANCE_TYPE,
        instance_count=args.instance_count,
        hyperparameters=hyperparameters,
        use_spot_instances=args.use_spot if args.use_spot is not None else Config.USE_SPOT_INSTANCES,
    )
    
    # Start training
    print("\nStarting training...")
    estimator = trainer.start_training(
        estimator=estimator,
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        wait=args.wait
    )
    
    print("\n✅ Training job launched successfully!")
    print(f"Job name: {estimator.latest_training_job.name}")
    print(f"Model artifacts will be saved to: {estimator.model_data}")
    
    # Deploy if requested
    if args.deploy:
        print("\n" + "=" * 50)
        print("Step 3: Deploying model")
        print("=" * 50)
        
        predictor = trainer.deploy_model(
            estimator,
            instance_type=args.inference_instance_type or Config.INFERENCE_INSTANCE_TYPE,
            initial_instance_count=1
        )
        
        print("\n✅ Model deployed successfully!")
        print(f"Endpoint name: {predictor.endpoint_name}")
    
    return estimator


def main():
    parser = argparse.ArgumentParser(
        description="CESCO sLLM Fine-tuning Pipeline with SageMaker and LoRA"
    )
    
    # Pipeline steps
    parser.add_argument("--prepare-data", action="store_true",
                        help="Prepare training data")
    parser.add_argument("--train", action="store_true",
                        help="Launch training job")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps (prepare data + train)")
    
    # AWS Configuration (optional - uses config.py defaults)
    parser.add_argument("--role", type=str,
                        help=f"SageMaker execution role ARN (default: from config.py)")
    parser.add_argument("--bucket", type=str,
                        help=f"S3 bucket name (default: from config.py)")
    parser.add_argument("--region", type=str,
                        help=f"AWS region (default: {Config.AWS_REGION})")
    
    # Model Configuration (optional - uses config.py defaults)
    parser.add_argument("--model-name", type=str,
                        help=f"Base model name (default: {Config.BASE_MODEL_NAME})")
    
    # Training Configuration (optional - uses config.py defaults)
    parser.add_argument("--instance-type", type=str,
                        help=f"Training instance type (default: {Config.TRAINING_INSTANCE_TYPE})")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of training instances (default: 1)")
    parser.add_argument("--use-spot", action="store_true", default=None,
                        help=f"Use spot instances for training (default: {Config.USE_SPOT_INSTANCES})")
    parser.add_argument("--wait", action="store_true", default=True,
                        help="Wait for training completion")
    
    # Hyperparameters (optional - uses config.py defaults)
    parser.add_argument("--epochs", type=int,
                        help=f"Number of training epochs (default: {Config.NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int,
                        help=f"Training batch size (default: {Config.BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float,
                        help=f"Learning rate (default: {Config.LEARNING_RATE})")
    parser.add_argument("--lora-r", type=int,
                        help=f"LoRA rank (default: {Config.LORA_R})")
    parser.add_argument("--lora-alpha", type=int,
                        help=f"LoRA alpha (default: {Config.LORA_ALPHA})")
    
    # Deployment
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy model after training")
    parser.add_argument("--inference-instance-type", type=str,
                        help=f"Inference instance type (default: {Config.INFERENCE_INSTANCE_TYPE})")
    
    # Display configuration
    parser.add_argument("--show-config", action="store_true",
                        help="Display current configuration and exit")
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        Config.display()
        return
    
    # Display banner
    print("\n" + "=" * 50)
    print("CESCO sLLM Fine-tuning Pipeline")
    print("Instance Type: ml.p4d.24xlarge")
    print("Method: LoRA (Low-Rank Adaptation)")
    print("=" * 50 + "\n")
    
    # Execute steps
    if args.all or args.prepare_data:
        prepare_data()
    
    if args.all or args.train:
        # Check if role and bucket are available (from args or config)
        role = args.role or Config.SAGEMAKER_ROLE_ARN
        bucket = args.bucket or Config.S3_BUCKET
        
        if not role or not bucket:
            print("\n❌ Error: SageMaker role and S3 bucket must be configured")
            print("Either:")
            print("  1. Update config.py with your AWS credentials, or")
            print("  2. Pass --role and --bucket arguments")
            print("\nExample:")
            print("  python main.py --train --role <role-arn> --bucket <s3-bucket>")
            sys.exit(1)
        
        launch_training(args)
    
    if not (args.all or args.prepare_data or args.train):
        parser.print_help()
        print("\nExample usage:")
        print("  # Show current configuration")
        print("  python main.py --show-config")
        print("\n  # Prepare data only")
        print("  python main.py --prepare-data")
        print("\n  # Launch training (uses config.py defaults)")
        print("  python main.py --train")
        print("\n  # Launch training with custom parameters")
        print("  python main.py --train --epochs 5 --batch-size 2")
        print("\n  # Run all steps with deployment")
        print("  python main.py --all --deploy")


if __name__ == "__main__":
    main()
