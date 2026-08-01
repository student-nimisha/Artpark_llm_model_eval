from data import load_dataset


def load_hf_dataset(config):
    """
    Streams a Hugging Face dataset.
    """

    dataset_name = config["dataset"]["name"]
    split = config["dataset"]["split"]
    streaming = config["dataset"]["streaming"]

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming
    )

    return dataset
