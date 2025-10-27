"""
Data preparation module for SageMaker fine-tuning.
Converts processed dataset to training format.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_categories_and_bugs():
    """Load available categories and bug types from assets."""
    categories_df = pd.read_csv('assets/voc_category_final.csv')
    bugs_df = pd.read_csv('assets/bugs_df.csv')
    
    # Format categories as "대분류__중분류__소분류"
    categories = []
    for _, row in categories_df.iterrows():
        if pd.notna(row.get('대분류')) and pd.notna(row.get('중분류')) and pd.notna(row.get('소분류')):
            category = f"{row['대분류']}__{row['중분류']}__{row['소분류']}"
            categories.append(category)
    
    # Get bug types
    bug_types = bugs_df['bug_name'].tolist() if 'bug_name' in bugs_df.columns else []
    
    return categories, bug_types


def create_prompt(input_text: str, categories: List[str], bug_types: List[str]) -> str:
    """Create instruction prompt for the model."""
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
- Do NOT use nested objects like {{"top": "...", "mid": "...", "bottom": "..."}}
- Use exact strings from the available categories list above
- Output ONLY valid JSON in this exact field order: claim_status, summary, bug_type, keywords, categories, evidences
- Stop immediately after the closing }}

CLAIM vs NON-CLAIM CLASSIFICATION:
- Use "claim" when: Customer expresses dissatisfaction, complaint, or problem with service/product
- Use "non-claim" when: Simple inquiries, requests for information, or positive feedback without complaints

Required JSON format:
{{
    "claim_status": "claim" or "non-claim", // claim=complaints/problems, non-claim=inquiries/requests
    "summary": "고객 이슈 요약 (Korean, max 1 sentence)",
    "bug_type": "bug/animal name from Available Bug Types list above OR null if no bugs mentioned",
    "keywords": ["키워드1", "키워드2", "키워드3"], // max 3 keywords
    "categories": ["대분류__중분류__소분류", "..."], // maximum 5 categories from available list
    "evidences": ["각 분류 결정에 대한 근거", "..."] // one evidence per category
}}

CRITICAL FIELD REQUIREMENTS:
- claim_status: Must be exactly "claim" or "non-claim" (not category names)
- bug_type: ONLY use if issue mentions actual bugs/animals from Available Bug Types list, otherwise use null
  (NOT for products like "정수기", services, or non-living things - use null for these)
- categories: Must be exact strings from Available Categories list above

### Input:
{input_text}

### Response:
"""
    return prompt


def create_response(data: Dict[str, Any]) -> str:
    """Create the expected JSON response."""
    response = {
        "claim_status": data.get("claim_status", "claim"),
        "summary": data.get("summary", ""),
        "bug_type": data.get("bug_type"),
        "keywords": data.get("keywords", []),
        "categories": data.get("categories", []),
        "evidences": data.get("evidences", [])
    }
    return json.dumps(response, ensure_ascii=False, indent=4)


def prepare_training_data(
    input_file: str,
    output_file: str,
    format_type: str = "jsonl"
) -> None:
    """
    Prepare training data in the format required for fine-tuning.
    
    Args:
        input_file: Path to processed dataset JSON
        output_file: Path to output file
        format_type: Output format ('jsonl' or 'json')
    """
    # Load processed data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load categories and bug types
    categories, bug_types = load_categories_and_bugs()
    
    training_samples = []
    
    for item in data:
        # Create prompt with instruction
        prompt = create_prompt(item['input'], categories, bug_types)
        
        # Create expected response
        response = create_response(item)
        
        # Format for instruction fine-tuning
        training_sample = {
            "instruction": prompt,
            "output": response
        }
        
        training_samples.append(training_sample)
    
    # Save in requested format
    if format_type == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dumps(training_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(training_samples)} training samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    # Prepare training data
    prepare_training_data(
        input_file="data/processed_train_dataset.json",
        output_file="data/train.jsonl",
        format_type="jsonl"
    )
    
    # Prepare test data
    prepare_training_data(
        input_file="data/processed_test_dataset.json",
        output_file="data/test.jsonl",
        format_type="jsonl"
    )
