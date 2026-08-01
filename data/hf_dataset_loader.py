from datasets import load_dataset


def load_hf_dataset(config):
    """
    Load the Hugging Face dataset using streaming.
    """

    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=config["dataset"]["streaming"]
    )

    return dataset


def preprocess_sample(sample, idx):
    """
    Convert every dataset into a common format.
    """

    return {

        "id": idx,

        "text": sample["text"],

        "label": sample["label"]

    }
