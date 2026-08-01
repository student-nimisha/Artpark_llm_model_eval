from data.hf_dataset_loader import (
    load_hf_dataset,
    preprocess_sample
)


def run_evaluation(config):

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = load_hf_dataset(config)

    print("\nDataset Loaded Successfully\n")

    print("Displaying first 5 processed samples:\n")

    for idx, sample in enumerate(dataset):

        sample = preprocess_sample(sample, idx)

        print(sample)

        print("-" * 60)

        if idx == 4:
            break
