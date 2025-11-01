import json

def main():
    with open("data/unsloth_test_dataset.json", "r") as f:
        test_data = json.load(f)
    
    train_test_split = 0.8
    split_index = int(len(test_data) * train_test_split)

    train_data = test_data[:split_index]
    val_data = test_data[split_index:]

    with open("data/unsloth_test_dataset_train_split.json", "w") as f:
        json.dump(train_data, f)

    with open("data/unsloth_test_dataset_test_split.json", "w") as f:
        json.dump(val_data, f)
    

    print(len(train_data))
    print(len(val_data))

    ## print the number of claim and non-claim in train_data
    claim_count = 0
    non_claim_count = 0
    for item in train_data:
        output = json.loads(item['output'])
        if output['is_claim'] == 'claim':
            claim_count += 1
        else:
            non_claim_count += 1
    print(f"Train split - Claim: {claim_count}, Non-claim: {non_claim_count}")
    ## print the number of claim and non-claim in val_data
    claim_count = 0
    non_claim_count = 0
    for item in val_data:
        output = json.loads(item['output'])
        if output['is_claim'] == 'claim':
            claim_count += 1
        else:
            non_claim_count += 1
    print(f"Test split - Claim: {claim_count}, Non-claim: {non_claim_count}")

if __name__ == "__main__":
    main()
