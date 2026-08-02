"""
Generic evaluation loop. Knows nothing about Gemma/Aya/Qwen/Llama — only
the BaseModel interface. Tracks language per row and produces a full
precision/recall/F1/confusion-matrix breakdown, overall and per language.

Output paths are always resolved to ABSOLUTE paths before writing, and
printed as absolute paths. This removes any ambiguity about where files
actually landed — a relative "outputs/..." path silently depends on the
current working directory, which is exactly what caused confusion before.
"""

import csv
import json
import os
from typing import Any, Dict

from metrics.metrics import compute_metrics, language_display_name
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
    print(f"[evaluator] file exists check: predictions={os.path.exists(pred_path)}, metrics={os.path.exists(metric_path)}")

    _print_summary(metrics)
    return metrics


def _print_block(name: str, block: Dict[str, Any]) -> None:
    if block.get("num_labeled", 0) == 0:
        print(f"\n--- {name} --- (no labeled samples)")
        return

    print(f"\n--- {name} ---")
    print(f"  samples: {block.get('num_samples')}   labeled: {block.get('num_labeled')}")
    print(f"  accuracy: {block.get('accuracy')}")
    print(
        f"  macro  -> precision: {block.get('macro_precision')}  "
        f"recall: {block.get('macro_recall')}  f1: {block.get('macro_f1')}"
    )
    print(
        f"  weighted -> precision: {block.get('weighted_precision')}  "
        f"recall: {block.get('weighted_recall')}  f1: {block.get('weighted_f1')}"
    )
    for label, pc in block.get("per_class", {}).items():
        print(
            f"    [{label:9s}] precision={pc['precision']}  recall={pc['recall']}  "
            f"f1={pc['f1']}  support={pc['support']}"
        )
    cm = block.get("confusion_matrix")
    if cm:
        print(f"  confusion matrix (rows=gold, cols=predicted), labels={cm['labels']}:")
        for row in cm["matrix"]:
            print(f"    {row}")


def _print_summary(metrics: Dict[str, Any]) -> None:
    print("\n===================== OVERALL =====================")
    _print_block("OVERALL", metrics["overall"])

    print("\n===================== PER LANGUAGE =====================")
    for lang_code, block in metrics["per_language"].items():
        name = language_display_name(lang_code)
        _print_block(f"{name} ({lang_code})", block)
