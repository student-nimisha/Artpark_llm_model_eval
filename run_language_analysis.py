"""
Runs the full Phase 2 (post-processing) pipeline for one model's saved
predictions: language detection -> per-language metrics -> plots ->
auto-push to GitHub.

No LLM inference happens anywhere in this script — it only reads the
predictions CSV that a completed `python main.py configs/<x>.json` run
already produced and pushed to GitHub.

Usage:
    python run_language_analysis.py gemma
    python run_language_analysis.py aya
    python run_language_analysis.py qwen3
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main(model_prefix: str) -> None:
    predictions_csv = f"outputs/{model_prefix}_predictions.csv"
    with_language_csv = f"outputs/{model_prefix}_predictions_with_language.csv"
    language_metrics_json = f"outputs/{model_prefix}_language_metrics.json"

    if not os.path.exists(predictions_csv):
        print(f"ERROR: {predictions_csv} not found. Run the model evaluation first:")
        print(f"       python main.py configs/{model_prefix}_config.json")
        sys.exit(1)

    run(["python", "analysis/add_language.py", predictions_csv])
    run(["python", "analysis/language_metrics.py", with_language_csv])
    run(["python", "analysis/plot_language_metrics.py", language_metrics_json])

    from utils.git_push import push_outputs_to_github

    files_to_push = [
        os.path.abspath(with_language_csv),
        os.path.abspath(language_metrics_json),
    ]
    plot_dir = os.path.join(REPO_ROOT, "outputs", "plots", model_prefix)
    if os.path.isdir(plot_dir):
        for fname in os.listdir(plot_dir):
            files_to_push.append(os.path.join(plot_dir, fname))

    push_outputs_to_github(
        repo_root=REPO_ROOT,
        file_paths=files_to_push,
        commit_message=f"Add per-language analysis for {model_prefix}",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_language_analysis.py <model_prefix>")
        print("Example: python run_language_analysis.py gemma")
        sys.exit(1)

    main(sys.argv[1])
