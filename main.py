"""
Entry point. Usage (Kaggle or shell):
    python main.py configs/gemma27b_kaggle_config.json
    python main.py configs/ayaexpanse_config.json

No default config path is assumed — you must always pass one explicitly.

This script forces the working directory to the folder main.py itself
lives in, so relative "outputs/..." and "configs/..." paths in your JSON
configs always resolve correctly regardless of the caller's cwd.

If GH_TOKEN and GH_REPO environment variables are set, output files are
automatically committed and pushed to GitHub after the run completes.
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
from utils.git_push import push_outputs_to_github


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

    output_cfg = config["output"]
    pred_path = os.path.abspath(output_cfg["prediction_file"])
    metric_path = os.path.abspath(output_cfg["metric_file"])

    push_outputs_to_github(
        repo_root=REPO_ROOT,
        file_paths=[pred_path, metric_path],
        commit_message=f"Auto: add {config['experiment_name']} evaluation results",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: you must specify a config file.")
        print("Usage: python main.py configs/gemma27b_kaggle_config.json")
        sys.exit(1)

    main(sys.argv[1])
