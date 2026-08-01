import json

from evaluator.evaluators.py import run_evaluation


CONFIG_PATH = "configs/gemma_config.json"


def load_config(config_path):

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


def main():

    config = load_config(CONFIG_PATH)

    run_evaluation(config)


if __name__ == "__main__":
    main()
