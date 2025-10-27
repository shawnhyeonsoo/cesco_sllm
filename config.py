"""
Configuration module for CESCO sLLM fine-tuning.
All configuration values are hardcoded for easy deployment.
"""
from pathlib import Path


class Config:
    """Configuration for SageMaker training."""
    
    # AWS Configuration
    SAGEMAKER_ROLE_ARN: str = "arn:aws:iam::326614947732:role/AmazonSageMakerFullAccess"
    S3_BUCKET: str = "amazon-sagemaker-326614947732-us-east-1-b6aed9d1f258"
    AWS_REGION: str = "us-east-1"
    
    # Model Configuration
    BASE_MODEL_NAME: str = "meta-llama/Llama-3.2-3B-Instruct"
    TRAINING_INSTANCE_TYPE: str = "ml.p4d.24xlarge"
    INFERENCE_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    
    # Training Hyperparameters
    NUM_EPOCHS: float = 0.1
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 2e-4
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: str = "q_proj,v_proj,k_proj,o_proj"
    
    # Data Configuration
    MAX_SEQ_LENGTH: int = 8092
    GRADIENT_ACCUMULATION_STEPS: int = 4
    WARMUP_STEPS: int = 100
    LOGGING_STEPS: int = 10
    SAVE_STEPS: int = 500
    EVAL_STEPS: int = 10
    
    # Optimization
    USE_SPOT_INSTANCES: bool = False
    FP16: bool = True
    
    # Weights & Biases
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "cesco-sllm-finetuning"
    
    # HuggingFace Token
    HUGGING_FACE_HUB_TOKEN: str = "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"
    
    # Weights & Biases API Key
    WANDB_API_KEY: str = "c81320346d825ecba691cbc52468fbb48b97e834"
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ASSETS_DIR: Path = PROJECT_ROOT / "assets"
    SRC_DIR: Path = PROJECT_ROOT / "src"
    
    @classmethod
    def get_hyperparameters(cls) -> dict:
        """Get training hyperparameters as dictionary."""
        return {
            "model_name": cls.BASE_MODEL_NAME,
            "max_seq_length": cls.MAX_SEQ_LENGTH,
            # LoRA parameters
            "lora_r": cls.LORA_R,
            "lora_alpha": cls.LORA_ALPHA,
            "lora_dropout": cls.LORA_DROPOUT,
            "lora_target_modules": cls.LORA_TARGET_MODULES,
            # Training parameters
            "num_train_epochs": cls.NUM_EPOCHS,
            "per_device_train_batch_size": cls.BATCH_SIZE,
            "per_device_eval_batch_size": cls.BATCH_SIZE,
            "gradient_accumulation_steps": cls.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": cls.LEARNING_RATE,
            "warmup_steps": cls.WARMUP_STEPS,
            "logging_steps": cls.LOGGING_STEPS,
            "save_steps": cls.SAVE_STEPS,
            "eval_steps": cls.EVAL_STEPS,
            "fp16": cls.FP16,
            # W&B logging
            "use_wandb": cls.USE_WANDB,
            "wandb_project": cls.WANDB_PROJECT
        }
    
    @classmethod
    def display(cls):
        """Display current configuration."""
        print("=" * 60)
        print("CESCO sLLM Training Configuration")
        print("=" * 60)
        print(f"\n📦 AWS Configuration:")
        print(f"  Role ARN: {cls.SAGEMAKER_ROLE_ARN}")
        print(f"  S3 Bucket: {cls.S3_BUCKET}")
        print(f"  Region: {cls.AWS_REGION}")
        
        print(f"\n🤖 Model Configuration:")
        print(f"  Base Model: {cls.BASE_MODEL_NAME}")
        print(f"  Training Instance: {cls.TRAINING_INSTANCE_TYPE}")
        print(f"  Inference Instance: {cls.INFERENCE_INSTANCE_TYPE}")
        
        print(f"\n🎯 Training Parameters:")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Max Sequence Length: {cls.MAX_SEQ_LENGTH}")
        
        print(f"\n🔧 LoRA Configuration:")
        print(f"  Rank (r): {cls.LORA_R}")
        print(f"  Alpha: {cls.LORA_ALPHA}")
        print(f"  Dropout: {cls.LORA_DROPOUT}")
        print(f"  Target Modules: {cls.LORA_TARGET_MODULES}")
        
        print(f"\n💰 Optimization:")
        print(f"  Use Spot Instances: {cls.USE_SPOT_INSTANCES}")
        print(f"  FP16 Training: {cls.FP16}")
        
        print(f"\n📊 Experiment Tracking:")
        print(f"  W&B Enabled: {cls.USE_WANDB}")
        if cls.USE_WANDB:
            print(f"  W&B Project: {cls.WANDB_PROJECT}")
        
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Display configuration when run directly
    Config.display()
