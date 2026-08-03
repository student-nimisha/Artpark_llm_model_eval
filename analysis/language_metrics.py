"""
Phase 2, Step 2 of the two-phase evaluation pipeline.

Takes a *_predictions_with_language.csv (produced by
analysis/add_language.py) and computes per-language classification
metrics (accuracy, precision/recall/F1, confusion matrix) by re-using
metrics/metrics.py's existing compute_metrics() function. No LLM
inference happens here — pure offline post-processing on saved output.

Usage:
    python analysis/language_metrics.py outputs/gemma_predictions_with_language.csv
"""

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.metrics import compute_metrics

MIN_SAMPLES_FOR_RELIABLE_METRICS = 20  # flag languages with very few examples


def load_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows or "language" not in rows[0]:
        raise ValueError(
            f"'{csv_path}' has no 'language' column. Run "
            "analysis/add_language.py on the original predictions CSV first."
        )
    return rows


def compute_per_language_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = compute_metrics(rows)

    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_lang[r.get("language", "unknown")].append(r)

    per_language = {}
    low_sample_warnings = []
    for lang, lang_rows in sorted(by_lang.items()):
        block = compute_metrics(lang_rows)
        block["num_rows"] = len(lang_rows)
        per_language[lang] = block
        if len(lang_rows) < MIN_SAMPLES_FOR_RELIABLE_METRICS:
            low_sample_warnings.append((lang, len(lang_rows)))

    return {
        "overall": overall,
        "per_language": per_language,
        "low_sample_languages": low_sample_warnings,
    }


def print_summary(results: Dict[str, Any]) -> None:
    overall = results["overall"]
    print("\n===================== OVERALL =====================")
    print(f"samples: {overall.get('num_samples')}   labeled: {overall.get('num_labeled')}")
    print(f"accuracy: {overall.get('accuracy')}   macro F1: {overall.get('macro_f1')}")

    print("\n===================== PER LANGUAGE =====================")
    for lang, block in results["per_language"].items():
        if block.get("num_labeled", 0) == 0:
            print(f"{lang}: no labeled samples")
            continue
        print(
            f"{lang:45s} n={block['num_rows']:5d}  acc={block.get('accuracy')}  "
            f"macro_f1={block.get('macro_f1')}"
        )
        for label, pc in block.get("per_class", {}).items():
            print(
                f"    [{label:9s}] precision={pc['precision']}  recall={pc['recall']}  "
                f"f1={pc['f1']}  support={pc['support']}"
            )

    if results["low_sample_languages"]:
        print("\n[WARNING] These languages have very few samples — treat their metrics with caution:")
        for lang, n in results["low_sample_languages"]:
            print(f"  {lang}: only {n} samples")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/language_metrics.py outputs/<model>_predictions_with_language.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    rows = load_rows(input_path)
    results = compute_per_language_metrics(rows)

    base = input_path.replace("_predictions_with_language.csv", "")
    output_path = f"{base}_language_metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[language_metrics] wrote -> {output_path}")
    print_summary(results)
