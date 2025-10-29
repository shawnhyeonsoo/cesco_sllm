"""
Prepare CESCO dataset for Unsloth training.
Converts JSON data to instruction-input-output format for Alpaca-style prompts.
"""

import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_json_data(file_path: str) -> list:
    """Load JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def convert_to_instruction_format(data: list) -> list:
    possible_category_df = pd.read_csv(
        PROJECT_ROOT / "assets" / "voc_category_final.csv"
    )
    possible_category_list = [
        f"{row['대분류']}__{row['중분류']}__{row['소분류']}"
        for _, row in possible_category_df.iterrows()
    ]

    possible_bug_df = pd.read_csv(PROJECT_ROOT / "assets" / "bugs_df.csv")
    possible_bug_types = [row["BugsNm"] for _, row in possible_bug_df.iterrows()]
    """
    Convert CESCO data to instruction-input-output format.
    
    Each example should have:
    - instruction: Task description
    - input: Customer complaint/input text
    - output: JSON response with analysis
    """
    instruction = """다음 고객의 민원 내용을 분석하여 JSON 형식으로 응답해주세요. 다음 정보를 포함해야 합니다:
1. claim_status: 클레임 여부 (claim 또는 non-claim)
2. summary: 민원 내용 요약
3. bug_type: 해충 종류 (해충 관련인 경우만, 없으면 null) (Choose one from {possible_bug_types})
4. keywords: 주요 키워드 리스트
5. categories: 분류 카테고리 리스트 (Choose maximum 5 from [{possible_categories}])
6. evidences: 근거가 되는 문장 리스트""".format(
        possible_bug_types=", ".join(possible_bug_types),
        possible_categories=", ".join(possible_category_list),
    )

    formatted_data = []

    for item in data:
        # Create output JSON
        output_dict = {
            "claim_status": item.get("claim_status", ""),
            "summary": item.get("summary", ""),
            "bug_type": item.get("bug_type"),
            "keywords": item.get("keywords", []),
            "categories": item.get("categories", []),
            "evidences": item.get("evidences", []),
        }

        if "기타__기타__기타" in output_dict["categories"]:
            output_dict["categories"].remove("기타__기타__기타")
            ## remove category with '기타' in any part
            output_dict["categories"] = [
                cat for cat in output_dict["categories"] if "기타" not in cat
            ]
            if not output_dict["categories"]:
                continue  # Skip if no valid categories left
        formatted_item = {
            "instruction": instruction,
            "input": item.get("input", ""),
            "output": json.dumps(output_dict, ensure_ascii=False, indent=2),
        }

        formatted_data.append(formatted_item)

    return formatted_data


def main():
    """Main data preparation workflow."""
    print("=" * 80)
    print("PREPARING DATA FOR UNSLOTH TRAINING")
    print("=" * 80)

    # Load original data
    train_file = PROJECT_ROOT / "data" / "processed_train_dataset.json"
    test_file = PROJECT_ROOT / "data" / "processed_test_dataset.json"

    print("\nLoading data from:")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")

    # Load and convert training data
    print("\n📊 Processing training data...")
    train_data = load_json_data(train_file)
    print(f"  Original: {len(train_data)} examples")

    formatted_train = convert_to_instruction_format(train_data)
    print(f"  Formatted: {len(formatted_train)} examples")

    # Save formatted training data
    output_train_file = PROJECT_ROOT / "data" / "unsloth_train_dataset.json"
    with open(output_train_file, "w", encoding="utf-8") as f:
        json.dump(formatted_train, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to: {output_train_file}")

    # Load and convert test data
    if test_file.exists():
        print("\n📊 Processing test data...")
        test_data = load_json_data(test_file)
        print(f"  Original: {len(test_data)} examples")

        formatted_test = convert_to_instruction_format(test_data)
        print(f"  Formatted: {len(formatted_test)} examples")

        # Save formatted test data
        output_test_file = PROJECT_ROOT / "data" / "unsloth_test_dataset.json"
        with open(output_test_file, "w", encoding="utf-8") as f:
            json.dump(formatted_test, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to: {output_test_file}")

    # Show example
    print("\n" + "=" * 80)
    print("EXAMPLE FORMATTED DATA")
    print("=" * 80)
    print("\nInstruction:")
    print(formatted_train[0]["instruction"])
    print("\nInput:")
    print(formatted_train[0]["input"][:200] + "...")
    print("\nOutput:")
    print(formatted_train[0]["output"][:300] + "...")
    print("=" * 80)

    print("\n✅ Data preparation complete!")
    print("\nFormatted files:")
    print(f"  - {output_train_file}")
    if test_file.exists():
        print(f"  - {output_test_file}")


if __name__ == "__main__":
    main()
