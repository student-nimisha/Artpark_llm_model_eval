"""
Entry point. Usage (Colab or shell):
    python main.py configs/gemma_config.json

Swapping models is a config-only change: point config["model"]["name"] at
a different Hugging Face model id.
"""

import json
import sys
from typing import Any, Dict

from data.hf_dataset_loader import load_streaming_dataset
from evaluator.evaluators import run_evaluation
from models.model_factory import load_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config_path: str) -> None:
    config = load_config(config_path)
    print(f"[main] experiment: {config['experiment_name']}")

    dataset = load_streaming_dataset(config)
    print("[main] dataset ready (streaming).")

    model = load_model(config)
    print(f"[main] model loaded: {config['model']['name']}")

    metrics = run_evaluation(config, model, dataset)
    print("[main] done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gemma_config.json"
    main(cfg_path)
