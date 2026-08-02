"""
Entry point. Usage (Kaggle or shell):
    python main.py configs/gemma_config.json
    python main.py configs/ayaexpanse_config.json

No default config path is assumed — you must always pass one explicitly.

IMPORTANT (Kaggle-specific): this script forces the working directory to
the folder main.py itself lives in, before doing anything else. That
means "outputs/..." and "configs/..." in your JSON configs always resolve
relative to the repo root, no matter which directory the notebook cell
was run from. This is what fixes "I can't find my output file."
"""

import json
import os
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)

from data.hf_dataset_loader import load_streaming_dataset
from evaluator.evaluators import run_evaluation
from models.model_factory import load_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config_path: str) -> None:
    config = load_config(config_path)
    print(f"[main] repo root: {REPO_ROOT}")
    print(f"[main] config file: {os.path.abspath(config_path)}")
    print(f"[main] experiment: {config['experiment_name']}")
    print(f"[main] model: {config['model']['name']}")

    dataset = load_streaming_dataset(config)
    print("[main] dataset ready (streaming).")

    model = load_model(config)
    print(f"[main] model loaded: {config['model']['name']}")

    metrics = run_evaluation(config, model, dataset)
    print("[main] done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: you must specify a config file.")
        print("Usage: python main.py configs/gemma_config.json")
        print("       python main.py configs/ayaexpanse_config.json")
        sys.exit(1)

    main(sys.argv[1])
