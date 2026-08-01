"""
Generic evaluation loop. Knows nothing about Gemma/Aya/Qwen/Llama — only
the BaseModel interface. Now also tracks language per row and produces a
full precision/recall/F1/confusion-matrix breakdown, overall and per
language, via metrics/metrics.py.
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

    pred_path = output_cfg["prediction_file"]
    metric_path = output_cfg["metric_file"]
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)

    rows = []

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        text = example["text"]
        language = example.get("language", "unknown") or "unknown"
        gold_label_raw = example.get("label")
        gold_label_norm = normalize_label(gold_label_raw) if gold_label_raw is not None else "unknown"

        messages = build_prompt(text)
        raw_output = model.generate(messages, gen_cfg)
        pred_label = normalize_label(raw_output)

        rows.append(
            {
                "text": text,
                "language": language,
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
            fieldnames=[
                "text",
                "language",
                "gold_label",
                "gold_label_norm",
                "raw_model_output",
                "predicted_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics = compute_metrics(rows)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluator] saved predictions -> {pred_path}")
    print(f"[evaluator] saved metrics -> {metric_path}")

    _print_summary(metrics)
    return metrics


def _print_summary(metrics: Dict[str, Any]) -> None:
    overall = metrics["overall"]
    print("\n===== OVERALL =====")
    print(f"samples: {overall.get('num_samples')}  labeled: {overall.get('num_labeled')}")
    print(f"accuracy: {overall.get('accuracy')}")
    print(f"macro F1: {overall.get('macro_f1')}   weighted F1: {overall.get('weighted_f1')}")
    for label, block in overall.get("per_class", {}).items():
        print(
            f"  [{label}] precision={block['precision']} "
            f"recall={block['recall']} f1={block['f1']} support={block['support']}"
        )

    print("\n===== PER LANGUAGE =====")
    for lang, block in metrics["per_language"].items():
        if block.get("num_labeled", 0) == 0:
            print(f"{lang}: no labeled samples")
            continue
        print(
            f"{lang}: n={block['num_samples']} acc={block['accuracy']} "
            f"macro_f1={block['macro_f1']}"
        )
