"""
Simple overall classification metrics for the profanity-detection task.
No per-language breakdown — the dataset doesn't provide a language field.
"""

from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["safe", "not safe"]


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: list of dicts, each with:
        "gold_label_norm": "safe" | "not safe" | "unknown"
        "predicted_label": "safe" | "not safe" | "unknown"
    """
    pairs = [
        (r["gold_label_norm"], r["predicted_label"])
        for r in rows
        if r["gold_label_norm"] in LABELS
    ]

    result: Dict[str, Any] = {"num_samples": len(rows), "num_labeled": len(pairs)}

    if not pairs:
        return result

    gold = [g for g, _ in pairs]
    pred = [p for _, p in pairs]

    acc = accuracy_score(gold, pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=LABELS, average="weighted", zero_division=0
    )

    cm = confusion_matrix(gold, pred, labels=LABELS).tolist()

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    result.update(
        {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class": per_class,
            "confusion_matrix": {"labels": LABELS, "matrix": cm},
        }
    )
    return result
