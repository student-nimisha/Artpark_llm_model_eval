"""
Generic evaluation loop with checkpoint/resume support.

Long runs can get interrupted (Kaggle session timeout, GPU quota limit,
a single pathological sample, etc.). Instead of losing all progress, we:
  1. Write each prediction row to the CSV immediately, flushing every
     20 samples.
  2. Track how many samples have been processed in a checkpoint file
     (predictions_file + ".checkpoint").
  3. On start, if a checkpoint exists, skip that many samples in the
     dataset stream and append to the existing CSV instead of
     overwriting it.
Re-running `python main.py <config>` after any interruption simply
continues instead of restarting from zero.
"""

import csv
import json
import os
from typing import Any, Dict

from metrics.metrics import compute_metrics
from prompts.profanity_prompt import build_prompt
from utils.postprocess import normalize_label

CSV_FIELDS = ["text", "gold_label", "gold_label_norm", "raw_model_output", "predicted_label"]


def _checkpoint_path(pred_path: str) -> str:
    return pred_path + ".checkpoint"


def _read_checkpoint(checkpoint_path: str) -> int:
    if not os.path.exists(checkpoint_path):
        return 0
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return int(content) if content else 0


def _write_checkpoint(checkpoint_path: str, count: int) -> None:
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write(str(count))


def run_evaluation(config: Dict[str, Any], model, dataset) -> Dict[str, Any]:
    gen_cfg = config.get("generation", {})
    output_cfg = config["output"]
    max_samples = config.get("evaluation", {}).get("max_samples")

    pred_path = os.path.abspath(output_cfg["prediction_file"])
    metric_path = os.path.abspath(output_cfg["metric_file"])
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metric_path) or ".", exist_ok=True)

    checkpoint_path = _checkpoint_path(pred_path)
    already_processed = _read_checkpoint(checkpoint_path)

    resuming = already_processed > 0 and os.path.exists(pred_path)
    if resuming:
        print(f"[evaluator] RESUMING from checkpoint: {already_processed} samples already done")
        csv_mode = "a"
        write_header = False
    else:
        already_processed = 0
        csv_mode = "w"
        write_header = True

    if already_processed > 0:
        dataset = dataset.skip(already_processed)

    csv_file = open(pred_path, csv_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    i = already_processed
    try:
        for example in dataset:
            if max_samples is not None and i >= max_samples:
                break

            text = example["text"]
            gold_label_raw = example.get("label")
            gold_label_norm = normalize_label(gold_label_raw) if gold_label_raw is not None else "unknown"

            messages = build_prompt(text)
            raw_output = model.generate(messages, gen_cfg)
            pred_label = normalize_label(raw_output)

            writer.writerow(
                {
                    "text": text,
                    "gold_label": gold_label_raw,
                    "gold_label_norm": gold_label_norm,
                    "raw_model_output": raw_output,
                    "predicted_label": pred_label,
                }
            )

            i += 1

            if i % 20 == 0:
                csv_file.flush()
                _write_checkpoint(checkpoint_path, i)
                print(f"[evaluator] processed {i} samples...")
    finally:
        csv_file.flush()
        csv_file.close()
        _write_checkpoint(checkpoint_path, i)

    print(f"[evaluator] total processed so far: {i}")

    rows = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    metrics = compute_metrics(rows)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[evaluator] saved predictions -> {pred_path}")
    print(f"[evaluator] saved metrics -> {metric_path}")

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
