import json


def main():
    with open("data/inference_results_on_train.json", "r") as f:
        inferred_train_data = json.load(f)
    inferred_train_data = inferred_train_data['results']
    with open("data/unsloth_train_dataset.json", "r") as f:
        original_train_data = json.load(f)
    
    original_train_data = original_train_data[:10]

    count = 0
    outputs = []
    for original, inferred in zip(original_train_data, inferred_train_data):
        prediction = inferred['prediction']

        generated = json.loads(original['output'])
        if len(prediction['categories']) == 0:
            prediction['categories'] = generated['categories']
        try:
            sample = {
                "instruction": original["instruction"],
                "input": original["input"],
                "output": json.dumps(
                    {
                        "is_claim": prediction["is_claim"],
                        "summary": generated['summary'],
                        "bug_type": generated['bug_type'],
                        "categories": prediction['categories'] if prediction["is_claim"] else generated['categories'],
                        "keywords": generated['keywords'],
                        "evidences": generated['evidences'][:len(prediction['categories'])] + [generated['evidences'][0]] * (len(prediction['categories']) - len(generated['evidences'])),
                    },
                    ensure_ascii=False),
            }
        except:
            count += 1
            sample = {
                "instruction": original["instruction"],
                "input": original["input"],
                "output": json.dumps(
                    {
                        "is_claim": False,
                        "summary": generated['summary'],
                        "bug_type": generated['bug_type'],
                        "categories": generated['categories'],
                        "keywords": generated['keywords'],
                        "evidences": generated['evidences'],
            }, ensure_ascii=False),
            }
        outputs.append(sample)
    with open("data/modified_unsloth_train.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(count)
    print(outputs[:5])

if __name__ == "__main__":
    main()