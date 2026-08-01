from data.hf_dataset_loader import load_hf_dataset


def run_evaluation(config):

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = load_hf_dataset(config)

    print("\nDataset Loaded Successfully\n")

    print("Displaying first 5 samples:\n")

    for i, sample in enumerate(dataset):

        print(f"Sample {i+1}")

        print("Text :", sample["text"])

        print("Label:", sample["label"])

        print("-" * 60)

        if i == 4:
            break
