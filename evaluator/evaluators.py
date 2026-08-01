"""
Generic evaluation loop. Knows nothing about Gemma/Aya/Qwen/Llama and
nothing about model internals — only the BaseModel interface
(model.generate(messages, generation_config) -> str).
"""

import csv
import json
import os
from typing import Any, Dict

from prompts.profanity_prompt import build_prompt
from utils.postprocess import normalize_label


def run_evaluation(config: Dict[str, Any], model, dataset) -> Dict[str, Any]:
    gen_cfg = config.get("generation", {})
    output_cfg = config["output"]
    max_samples = config.get("evaluation", {}).get("max_samples")

    pred_path = output_cfg["prediction_file"]
    metric_path = output_cfg["metric_file"]
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)

    correct = 0
    total_labeled = 0
    rows = []

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        text = example["text"]
        gold_label_raw = example.get("label")

        messages = build_prompt(text)
        raw_output = model.generate(messages, gen_cfg)
        pred_label = normalize_label(raw_output)

        row = {
            "text": text,
            "gold_label": gold_label_raw,
            "raw_model_output": raw_output,
            "predicted_label": pred_label,
        }
        rows.append(row)

        if gold_label_raw is not None:
            gold_label = normalize_label(gold_label_raw)
            total_labeled += 1
            if gold_label == pred_label:
                correct += 1

        if (i + 1) % 20 == 0:
            print(f"[evaluator] processed {i + 1} samples...")

    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "gold_label", "raw_model_output", "predicted_label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics: Dict[str, Any] = {"num_samples": len(rows)}
    if total_labeled > 0:
        metrics["accuracy"] = correct / total_labeled
        metrics["num_labeled"] = total_labeled

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluator] saved predictions -> {pred_path}")
    print(f"[evaluator] saved metrics -> {metric_path}")
    return metrics
