"""
Generic evaluation loop. Knows nothing about Gemma/Aya/Qwen/Llama — only
the BaseModel interface (model.generate(messages, generation_config) -> str).
No language tracking — overall metrics only.
"""

import csv
import json
import os
from typing import Any, Dict

from metrics.metrics import compute_metrics
from prompts.profanity_prompt import build_prompt
from utils.postprocess import normalize_label


def run_evaluation(config: Dict[str, Any], model, dataset) -> Dict[str, Any]:
    gen_cfg = config.get("generation", {})
    output_cfg = config["output"]
    max_samples = config.get("evaluation", {}).get("max_samples")

    pred_path = os.path.abspath(output_cfg["prediction_file"])
    metric_path = os.path.abspath(output_cfg["metric_file"])
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metric_path) or ".", exist_ok=True)

    rows = []

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        text = example["text"]
        gold_label_raw = example.get("label")
        gold_label_norm = normalize_label(gold_label_raw) if gold_label_raw is not None else "unknown"

        messages = build_prompt(text)
        raw_output = model.generate(messages, gen_cfg)
        pred_label = normalize_label(raw_output)

        rows.append(
            {
                "text": text,
                "gold_label": gold_label_raw,
                "gold_label_norm": gold_label_norm,
                "raw_model_output": raw_output,
                "predicted_label": pred_label,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"[evaluator] processed {i + 1} samples...")

    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "gold_label", "gold_label_norm", "raw_model_output", "predicted_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = compute_metrics(rows)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluator] saved predictions -> {pred_path}")
    print(f"[evaluator] saved metrics -> {metric_path}")
    print(f"[evaluator] file exists check: predictions={os.path.exists(pred_path)}, metrics={os.path.exists(metric_path)}")

    _print_summary(metrics)
    return metrics


def _print_summary(metrics: Dict[str, Any]) -> None:
    print("\n===================== RESULTS =====================")
    print(f"samples: {metrics.get('num_samples')}   labeled: {metrics.get('num_labeled')}")
    if metrics.get("num_labeled", 0) == 0:
        print("No labeled samples to score.")
        return

    print(f"accuracy: {metrics.get('accuracy')}")
    print(
        f"macro  -> precision: {metrics.get('macro_precision')}  "
        f"recall: {metrics.get('macro_recall')}  f1: {metrics.get('macro_f1')}"
    )
    print(
        f"weighted -> precision: {metrics.get('weighted_precision')}  "
        f"recall: {metrics.get('weighted_recall')}  f1: {metrics.get('weighted_f1')}"
    )
    for label, pc in metrics.get("per_class", {}).items():
        print(
            f"  [{label:9s}] precision={pc['precision']}  recall={pc['recall']}  "
            f"f1={pc['f1']}  support={pc['support']}"
        )
    cm = metrics.get("confusion_matrix")
    if cm:
        print(f"confusion matrix (rows=gold, cols=predicted), labels={cm['labels']}:")
        for row in cm["matrix"]:
            print(f"  {row}")
