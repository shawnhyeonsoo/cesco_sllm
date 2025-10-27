"""
Inference module for the fine-tuned model.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sagemaker
from sagemaker.huggingface import HuggingFacePredictor
import pandas as pd


class LocalInference:
    """Local inference with fine-tuned LoRA model."""
    
    def __init__(
        self,
        base_model_name: str,
        lora_weights_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize local inference.
        
        Args:
            base_model_name: Base model name
            lora_weights_path: Path to LoRA weights
            device: Device for inference
        """
        print(f"Loading model on {device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.base_model,
            lora_weights_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        self.model.eval()
        self.device = device
        
        print("Model loaded successfully!")
    
    def create_prompt(self, input_text: str) -> str:
        """Create prompt from input text."""
        # Load categories and bug types
        categories_df = pd.read_csv('assets/voc_category_final.csv')
        bugs_df = pd.read_csv('assets/bugs_df.csv')
        
        categories = []
        for _, row in categories_df.iterrows():
            if pd.notna(row.get('대분류')) and pd.notna(row.get('중분류')) and pd.notna(row.get('소분류')):
                category = f"{row['대분류']}__{row['중분류']}__{row['소분류']}"
                categories.append(category)
        
        bug_types = bugs_df['bug_name'].tolist() if 'bug_name' in bugs_df.columns else []
        
        existing_categories = "\n".join([f"- {cat}" for cat in categories])
        possible_bug_types = "\n".join([f"- {bug}" for bug in bug_types])
        
        prompt = f"""### Instruction:
You are an expert in classifying customer VOC issues. Classify the customer issue into appropriate categories from the available list and provide structured JSON output.

Available Categories (format: 대분류__중분류__소분류):
{existing_categories}

Available Bug Types:
{possible_bug_types}

IMPORTANT FORMAT REQUIREMENTS:
- Categories MUST be in format: ["대분류__중분류__소분류", "대분류__중분류__소분류"]  
- Maximum 5 categories allowed
- Provide one evidence for each selected category
- Output ONLY valid JSON

Required JSON format:
{{
    "claim_status": "claim" or "non-claim",
    "summary": "고객 이슈 요약",
    "bug_type": "bug name or null",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "categories": ["대분류__중분류__소분류", "..."],
    "evidences": ["각 분류 결정에 대한 근거", "..."]
}}

### Input:
{input_text}

### Response:
"""
        return prompt
    
    def predict(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> dict:
        """
        Generate prediction for input text.
        
        Args:
            input_text: Customer input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with predicted fields
        """
        # Create prompt
        prompt = self.create_prompt(input_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find the response section
            response_start = generated_text.find("### Response:")
            if response_start != -1:
                json_text = generated_text[response_start + len("### Response:"):].strip()
                
                # Try to parse JSON
                # Find first { and last }
                start_idx = json_text.find("{")
                end_idx = json_text.rfind("}") + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = json_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Generated text: {generated_text}")
        
        return {
            "claim_status": "claim",
            "summary": "",
            "bug_type": None,
            "keywords": [],
            "categories": [],
            "evidences": [],
            "raw_output": generated_text
        }


class SageMakerInference:
    """SageMaker endpoint inference."""
    
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        """
        Initialize SageMaker inference.
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region
        """
        self.predictor = HuggingFacePredictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker.Session()
        )
        print(f"Connected to endpoint: {endpoint_name}")
    
    def predict(self, input_text: str, **kwargs) -> dict:
        """
        Generate prediction using SageMaker endpoint.
        
        Args:
            input_text: Customer input text
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with predicted fields
        """
        # Create prompt (similar to LocalInference)
        # For brevity, using simplified version
        prompt = f"### Input:\n{input_text}\n### Response:\n"
        
        payload = {
            "inputs": prompt,
            "parameters": kwargs
        }
        
        response = self.predictor.predict(payload)
        
        # Parse response
        try:
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get("generated_text", "")
                
                # Extract JSON
                start_idx = generated_text.find("{")
                end_idx = generated_text.rfind("}") + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return response


def test_inference():
    """Test inference with sample inputs."""
    # Example usage
    print("Testing inference...")
    
    # Test data
    test_input = "정수기에서 물이 안 나오고 소음이 발생합니다. A/S 요청드립니다."
    
    # For local testing (after training)
    # inference = LocalInference(
    #     base_model_name="meta-llama/Llama-2-7b-hf",
    #     lora_weights_path="/path/to/lora/weights"
    # )
    # result = inference.predict(test_input)
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # For SageMaker endpoint
    # inference = SageMakerInference(endpoint_name="your-endpoint-name")
    # result = inference.predict(test_input)
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("Inference test completed!")


if __name__ == "__main__":
    test_inference()
