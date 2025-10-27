"""
Example notebook for CESCO sLLM fine-tuning workflow.
"""

# %% [markdown]
# # CESCO sLLM Fine-tuning with SageMaker and LoRA
# 
# This notebook demonstrates the complete workflow for fine-tuning a small language model
# for customer VOC classification using AWS SageMaker with LoRA on ml.p4d.24xlarge instances.

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.append('../src')

import json
import pandas as pd
from pathlib import Path

# %% [markdown]
# ## 1. Data Exploration

# %%
# Load training data
with open('../data/processed_train_dataset.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"Training samples: {len(train_data)}")
print("\nSample data:")
print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

# %%
# Load categories
categories_df = pd.read_csv('../assets/voc_category_final.csv')
print(f"\nAvailable categories: {len(categories_df)}")
print(categories_df.head())

# %%
# Analyze category distribution
all_categories = []
for item in train_data:
    all_categories.extend(item['categories'])

category_counts = pd.Series(all_categories).value_counts()
print("\nTop 10 categories:")
print(category_counts.head(10))

# %%
# Analyze claim vs non-claim
claim_counts = pd.Series([item['claim_status'] for item in train_data]).value_counts()
print("\nClaim status distribution:")
print(claim_counts)

# %% [markdown]
# ## 2. Data Preparation

# %%
from data_preparation import prepare_training_data, load_categories_and_bugs, create_prompt

# Prepare training data
prepare_training_data(
    input_file='../data/processed_train_dataset.json',
    output_file='../data/train.jsonl',
    format_type='jsonl'
)

prepare_training_data(
    input_file='../data/processed_test_dataset.json',
    output_file='../data/test.jsonl',
    format_type='jsonl'
)

print("✅ Data preparation completed!")

# %%
# View prepared training sample
with open('../data/train.jsonl', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())

print("Instruction format:")
print(sample['instruction'][:500] + "...")
print("\nExpected output:")
print(sample['output'])

# %% [markdown]
# ## 3. AWS Configuration

# %%
import boto3
import sagemaker

# Check AWS credentials
try:
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    print(f"✅ AWS Account: {identity['Account']}")
    print(f"✅ User/Role: {identity['Arn']}")
except Exception as e:
    print(f"❌ AWS credentials not configured: {e}")

# %%
# SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
region = session.boto_region_name

print(f"SageMaker Role: {role}")
print(f"S3 Bucket: {bucket}")
print(f"Region: {region}")

# %% [markdown]
# ## 4. Launch Training Job

# %%
from sagemaker_training import SageMakerTrainer

# Initialize trainer
trainer = SageMakerTrainer(
    role_arn=role,
    s3_bucket=bucket,
    region=region
)

# %%
# Upload data to S3
train_s3_uri, test_s3_uri = trainer.upload_data_to_s3(
    train_data_path='../data/train.jsonl',
    test_data_path='../data/test.jsonl'
)

print(f"✅ Data uploaded to S3")

# %%
# Configure hyperparameters
hyperparameters = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_wandb": True,
    "wandb_project": "cesco-sllm-finetuning"
}

print("Training configuration:")
print(json.dumps(hyperparameters, indent=2))

# %%
# Create training job
estimator = trainer.create_training_job(
    train_s3_uri=train_s3_uri,
    test_s3_uri=test_s3_uri,
    instance_type="ml.p4d.24xlarge",
    instance_count=1,
    hyperparameters=hyperparameters,
    use_spot_instances=False,  # Set to True for cost savings
)

print("✅ Training job configured")

# %%
# Start training (this will take several hours)
# Set wait=False to not block the notebook
estimator = trainer.start_training(
    estimator=estimator,
    train_s3_uri=train_s3_uri,
    test_s3_uri=test_s3_uri,
    wait=False,  # Set to True to wait for completion
    logs=True
)

print(f"✅ Training job started: {estimator.latest_training_job.name}")
print(f"\nMonitor training:")
print(f"  - AWS Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs")
print(f"  - Job name: {estimator.latest_training_job.name}")

# %% [markdown]
# ## 5. Monitor Training (Optional)
# 
# You can monitor training progress while it runs:

# %%
# Check training job status
import time

def check_training_status(estimator):
    """Check current training job status."""
    job_name = estimator.latest_training_job.name
    sm_client = boto3.client('sagemaker')
    
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    
    status = response['TrainingJobStatus']
    print(f"Job Status: {status}")
    
    if 'SecondaryStatus' in response:
        print(f"Secondary Status: {response['SecondaryStatus']}")
    
    if 'TrainingStartTime' in response:
        print(f"Start Time: {response['TrainingStartTime']}")
    
    if status == 'InProgress' and 'TrainingJobAnalytics' in response:
        analytics = response['TrainingJobAnalytics']
        print(f"Analytics: {analytics}")
    
    return status

# Check status (run this cell multiple times to monitor)
# status = check_training_status(estimator)

# %% [markdown]
# ## 6. Model Deployment (After Training Completes)

# %%
# Deploy model to endpoint (uncomment after training completes)
# predictor = trainer.deploy_model(
#     estimator,
#     instance_type="ml.g5.2xlarge",
#     initial_instance_count=1
# )
# print(f"✅ Model deployed to endpoint: {predictor.endpoint_name}")

# %% [markdown]
# ## 7. Inference Testing

# %%
# Test inference with deployed endpoint (uncomment after deployment)
# from inference import SageMakerInference
# 
# inference = SageMakerInference(endpoint_name=predictor.endpoint_name)
# 
# test_input = "정수기에서 물이 안 나오고 소음이 발생합니다."
# result = inference.predict(test_input)
# 
# print("Input:", test_input)
# print("\nPrediction:")
# print(json.dumps(result, ensure_ascii=False, indent=2))

# %%
# Test multiple inputs
# test_cases = [
#     "바퀴벌레가 주방에 나왔어요. 방역 요청합니다.",
#     "정수기 필터 교체 시기가 언제인가요?",
#     "서비스 비용이 생각보다 많이 나왔는데 확인 부탁드립니다."
# ]
# 
# for test_input in test_cases:
#     print(f"\n{'='*60}")
#     print(f"Input: {test_input}")
#     result = inference.predict(test_input)
#     print(f"\nCategories: {result.get('categories', [])}")
#     print(f"Claim Status: {result.get('claim_status', 'N/A')}")
#     print(f"Summary: {result.get('summary', 'N/A')}")

# %% [markdown]
# ## 8. Cost Analysis

# %%
# Estimate training cost
def estimate_training_cost(
    instance_type: str,
    num_instances: int,
    training_hours: float,
    use_spot: bool = False
):
    """Estimate training cost."""
    # Approximate on-demand pricing (us-east-1)
    pricing = {
        "ml.p4d.24xlarge": 32.77,
        "ml.g5.12xlarge": 5.67,
        "ml.g5.2xlarge": 1.19,
    }
    
    hourly_rate = pricing.get(instance_type, 0)
    
    if use_spot:
        hourly_rate *= 0.3  # ~70% discount for spot
    
    total_cost = hourly_rate * num_instances * training_hours
    
    print(f"Instance Type: {instance_type}")
    print(f"Instances: {num_instances}")
    print(f"Training Hours: {training_hours}")
    print(f"Hourly Rate: ${hourly_rate:.2f}")
    print(f"Spot Instances: {use_spot}")
    print(f"\nEstimated Total Cost: ${total_cost:.2f}")
    
    return total_cost

# Example: 3 epochs, estimated 4 hours
estimate_training_cost(
    instance_type="ml.p4d.24xlarge",
    num_instances=1,
    training_hours=4,
    use_spot=False
)

# %% [markdown]
# ## 9. Cleanup

# %%
# Delete endpoint (to avoid ongoing charges)
# if predictor:
#     predictor.delete_endpoint()
#     print("✅ Endpoint deleted")

# %%
# List and cleanup old training jobs
# sm_client = boto3.client('sagemaker')
# response = sm_client.list_training_jobs(
#     SortBy='CreationTime',
#     SortOrder='Descending',
#     MaxResults=10
# )
# 
# print("Recent training jobs:")
# for job in response['TrainingJobSummaries']:
#     print(f"  - {job['TrainingJobName']}: {job['TrainingJobStatus']}")

# %% [markdown]
# ## Summary
# 
# This notebook demonstrated:
# 1. ✅ Data exploration and preparation
# 2. ✅ AWS SageMaker configuration
# 3. ✅ Training job launch on ml.p4d.24xlarge with LoRA
# 4. ✅ Model deployment and inference
# 5. ✅ Cost estimation and cleanup
# 
# For production use:
# - Enable W&B for experiment tracking
# - Use spot instances for cost savings
# - Implement automated evaluation pipelines
# - Set up continuous training workflows
