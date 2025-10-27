"""
SageMaker training job orchestration script.
Launch fine-tuning job on ml.p4d.24xlarge with LoRA.
"""
import os
import json
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.s3 import S3Uploader
from pathlib import Path
import boto3
from datetime import datetime


class SageMakerTrainer:
    """Orchestrate SageMaker training jobs for sLLM fine-tuning."""
    
    def __init__(
        self,
        role_arn: str = None,
        s3_bucket: str = None,
        region: str = "us-east-1"
    ):
        """
        Initialize SageMaker trainer.
        
        Args:
            role_arn: IAM role ARN for SageMaker execution
            s3_bucket: S3 bucket name for data and model artifacts
            region: AWS region
        """
        self.session = sagemaker.Session(boto3.Session(region_name=region))
        self.region = region
        
        # Get role and bucket
        self.role = role_arn or sagemaker.get_execution_role()
        self.bucket = s3_bucket or self.session.default_bucket()
        
        print(f"SageMaker Role: {self.role}")
        print(f"S3 Bucket: {self.bucket}")
        print(f"Region: {self.region}")
    
    def upload_data_to_s3(
        self,
        train_data_path: str,
        test_data_path: str,
        s3_prefix: str = "cesco-sllm/data"
    ) -> tuple:
        """
        Upload training and test data to S3.
        
        Args:
            train_data_path: Local path to training data
            test_data_path: Local path to test data
            s3_prefix: S3 prefix for data upload
            
        Returns:
            Tuple of (train_s3_uri, test_s3_uri)
        """
        print("Uploading data to S3...")
        
        train_s3_uri = S3Uploader.upload(
            local_path=train_data_path,
            desired_s3_uri=f"s3://{self.bucket}/{s3_prefix}/train",
            sagemaker_session=self.session
        )
        
        test_s3_uri = S3Uploader.upload(
            local_path=test_data_path,
            desired_s3_uri=f"s3://{self.bucket}/{s3_prefix}/test",
            sagemaker_session=self.session
        )
        
        print(f"Train data uploaded to: {train_s3_uri}")
        print(f"Test data uploaded to: {test_s3_uri}")
        
        return train_s3_uri, test_s3_uri
    
    def create_training_job(
        self,
        train_s3_uri: str,
        test_s3_uri: str,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        instance_type: str = "ml.p4d.24xlarge",
        instance_count: int = 1,
        job_name: str = None,
        hyperparameters: dict = None,
        use_spot_instances: bool = False,
        max_run_time: int = 86400,  # 24 hours
        checkpoint_s3_uri: str = None
    ) -> HuggingFace:
        """
        Create and configure SageMaker training job.
        
        Args:
            train_s3_uri: S3 URI for training data
            test_s3_uri: S3 URI for test data
            model_name: Pretrained model name
            instance_type: SageMaker instance type (ml.p4d.24xlarge)
            instance_count: Number of instances
            job_name: Training job name
            hyperparameters: Additional hyperparameters
            use_spot_instances: Use spot instances for cost savings
            max_run_time: Maximum training time in seconds
            checkpoint_s3_uri: S3 URI for checkpoints
            
        Returns:
            HuggingFace estimator
        """
        # Generate job name if not provided
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"cesco-sllm-lora-{timestamp}"
        
        # Default hyperparameters
        default_hyperparameters = {
            "model_name": model_name,
            "max_seq_length": 2048,
            # LoRA parameters
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj,k_proj,o_proj",
            # Training parameters
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "fp16": True,
            # W&B logging
            "use_wandb": True,
            "wandb_project": "cesco-sllm-finetuning"
        }
        
        # Update with custom hyperparameters
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        # Configure checkpoint location
        if checkpoint_s3_uri is None:
            checkpoint_s3_uri = f"s3://{self.bucket}/cesco-sllm/checkpoints/{job_name}"
        
        # Create HuggingFace estimator
        print(f"Creating training job: {job_name}")
        print(f"Instance type: {instance_type}")
        print(f"Instance count: {instance_count}")
        
        huggingface_estimator = HuggingFace(
            entry_point="training_script.py",
            source_dir="src",
            instance_type=instance_type,
            instance_count=instance_count,
            role=self.role,
            transformers_version="4.49.0",
            pytorch_version="2.5.1",
            py_version="py311",
            hyperparameters=default_hyperparameters,
            dependencies=["src/requirements.txt"],
            use_spot_instances=use_spot_instances,
            max_run=max_run_time,
            max_wait=max_run_time + 3600 if use_spot_instances else None,
            checkpoint_s3_uri=checkpoint_s3_uri,
            environment={
                "NCCL_DEBUG": "INFO",
                "NCCL_SOCKET_IFNAME": "eth0",
                "WANDB_API_KEY": "c81320346d825ecba691cbc52468fbb48b97e834",
                "HUGGING_FACE_HUB_TOKEN": "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",
            },
            keep_alive_period_in_seconds=1800,  # Keep instance warm for 30 min
            volume_size=256,  # GB
            base_job_name="cesco-sllm-lora",
            tags=[
                {"Key": "Project", "Value": "CESCO-sLLM"},
                {"Key": "ModelType", "Value": "LoRA-FineTuning"}
            ]
        )
        
        return huggingface_estimator
    
    def start_training(
        self,
        estimator: HuggingFace,
        train_s3_uri: str,
        test_s3_uri: str,
        wait: bool = True,
        logs: bool = True
    ):
        """
        Start the training job.
        
        Args:
            estimator: HuggingFace estimator
            train_s3_uri: S3 URI for training data
            test_s3_uri: S3 URI for test data
            wait: Wait for job completion
            logs: Display training logs
        """
        print("Starting training job...")
        
        estimator.fit(
            inputs={
                "train": train_s3_uri,
                "test": test_s3_uri
            },
            wait=wait,
            logs=logs
        )
        
        print(f"Training job name: {estimator.latest_training_job.name}")
        print(f"Model artifacts: {estimator.model_data}")
        
        return estimator
    
    def deploy_model(
        self,
        estimator: HuggingFace,
        instance_type: str = "ml.g5.2xlarge",
        initial_instance_count: int = 1,
        endpoint_name: str = None
    ):
        """
        Deploy the trained model as a SageMaker endpoint.
        
        Args:
            estimator: Trained HuggingFace estimator
            instance_type: Instance type for inference
            initial_instance_count: Number of instances
            endpoint_name: Name for the endpoint
            
        Returns:
            Deployed predictor
        """
        if endpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            endpoint_name = f"cesco-sllm-endpoint-{timestamp}"
        
        print(f"Deploying model to endpoint: {endpoint_name}")
        
        predictor = estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        print(f"Endpoint deployed: {endpoint_name}")
        return predictor


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--role", type=str, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Pretrained model name")
    parser.add_argument("--instance-type", type=str, default="ml.p4d.24xlarge",
                        help="Instance type for training")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances")
    parser.add_argument("--use-spot", action="store_true",
                        help="Use spot instances")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy model after training")
    parser.add_argument("--wait", action="store_true", default=True,
                        help="Wait for training completion")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SageMakerTrainer(
        role_arn=args.role,
        s3_bucket=args.bucket,
        region=args.region
    )
    
    # Upload data to S3
    train_s3_uri, test_s3_uri = trainer.upload_data_to_s3(
        train_data_path="data/train.jsonl",
        test_data_path="data/test.jsonl"
    )
    
    # Create training job
    estimator = trainer.create_training_job(
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        model_name=args.model_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        use_spot_instances=args.use_spot,
    )
    
    # Start training
    estimator = trainer.start_training(
        estimator=estimator,
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        wait=args.wait
    )
    
    # Deploy if requested
    if args.deploy:
        predictor = trainer.deploy_model(estimator)
        print(f"Model deployed successfully!")
        
        # Test prediction
        test_input = {
            "inputs": "### Instruction:\nClassify this customer issue...\n### Input:\n정수기에서 물이 안 나옵니다.\n### Response:\n"
        }
        result = predictor.predict(test_input)
        print(f"Test prediction: {result}")


# Convenience functions for use in other scripts
def launch_training_job(
    train_data_path: str = "data/train.jsonl",
    test_data_path: str = "data/test.jsonl",
    role_arn: str = None,
    s3_bucket: str = None,
    region: str = "us-east-1",
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    instance_type: str = "ml.p4d.24xlarge",
    hyperparameters: dict = None,
    use_spot_instances: bool = False
) -> str:
    """
    Launch a SageMaker training job (convenience function).
    
    Returns:
        Training job name
    """
    # Initialize trainer
    trainer = SageMakerTrainer(
        role_arn=role_arn,
        s3_bucket=s3_bucket,
        region=region
    )
    
    # Upload data to S3
    train_s3_uri, test_s3_uri = trainer.upload_data_to_s3(
        train_data_path=train_data_path,
        test_data_path=test_data_path
    )
    
    # Create training job
    estimator = trainer.create_training_job(
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        model_name=model_name,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        use_spot_instances=use_spot_instances
    )
    
    # Start training (don't wait)
    trainer.start_training(
        estimator=estimator,
        train_s3_uri=train_s3_uri,
        test_s3_uri=test_s3_uri,
        wait=False,
        logs=False
    )
    
    return estimator.latest_training_job.name


def wait_for_training_job(job_name: str, region: str = "us-east-1"):
    """
    Wait for a training job to complete.
    
    Args:
        job_name: Training job name
        region: AWS region
    """
    session = sagemaker.Session(boto3.Session(region_name=region))
    
    print(f"Waiting for training job '{job_name}' to complete...")
    print("This may take a while. You can monitor progress in the AWS Console.")
    
    # Use waiter to wait for completion
    sm_client = session.sagemaker_client
    waiter = sm_client.get_waiter('training_job_completed_or_stopped')
    
    try:
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={
                'Delay': 60,  # Check every 60 seconds
                'MaxAttempts': 1440  # Max 24 hours (1440 minutes)
            }
        )
        
        # Get final status
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        if status == 'Completed':
            print(f"✅ Training job completed successfully!")
            return True
        else:
            print(f"⚠️ Training job ended with status: {status}")
            if 'FailureReason' in response:
                print(f"Failure reason: {response['FailureReason']}")
            return False
            
    except Exception as e:
        print(f"❌ Error waiting for training job: {e}")
        return False


def deploy_model(
    training_job_name: str,
    region: str = "us-east-1",
    instance_type: str = "ml.g5.2xlarge",
    initial_instance_count: int = 1,
    endpoint_name: str = None
) -> str:
    """
    Deploy a trained model to a SageMaker endpoint.
    
    Args:
        training_job_name: Name of completed training job
        region: AWS region
        instance_type: Instance type for inference
        initial_instance_count: Number of instances
        endpoint_name: Name for the endpoint
        
    Returns:
        Endpoint name
    """
    session = sagemaker.Session(boto3.Session(region_name=region))
    sm_client = session.sagemaker_client
    
    # Get training job details
    response = sm_client.describe_training_job(TrainingJobName=training_job_name)
    model_data_url = response['ModelArtifacts']['S3ModelArtifacts']
    role = response['RoleArn']
    
    # Generate endpoint name if not provided
    if endpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"cesco-sllm-endpoint-{timestamp}"
    
    # Create model
    model_name = f"{training_job_name}-model"
    
    try:
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': response['AlgorithmSpecification']['TrainingImage'],
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url
                }
            },
            ExecutionRoleArn=role
        )
        print(f"✅ Model created: {model_name}")
    except sm_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e) and 'already exists' in str(e):
            print(f"Model {model_name} already exists, using existing model")
        else:
            raise
    
    # Create endpoint configuration
    endpoint_config_name = f"{endpoint_name}-config"
    
    try:
        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': initial_instance_count,
                'InitialVariantWeight': 1.0
            }]
        )
        print(f"✅ Endpoint configuration created: {endpoint_config_name}")
    except sm_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e) and 'already exists' in str(e):
            print(f"Endpoint config {endpoint_config_name} already exists")
        else:
            raise
    
    # Create endpoint
    print(f"Creating endpoint: {endpoint_name}")
    print("This may take 5-10 minutes...")
    
    try:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        # Wait for endpoint to be in service
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"✅ Endpoint deployed successfully: {endpoint_name}")
        return endpoint_name
        
    except sm_client.exceptions.ClientError as e:
        if 'ValidationException' in str(e) and 'already exists' in str(e):
            print(f"Endpoint {endpoint_name} already exists")
            return endpoint_name
        else:
            raise


if __name__ == "__main__":
    main()
