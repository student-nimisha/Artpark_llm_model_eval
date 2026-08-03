"""
Phase 2, Step 3 (optional) of the two-phase evaluation pipeline.

Generates plots from a *_language_metrics.json file produced by
analysis/language_metrics.py:
  - a bar chart of accuracy and macro-F1 per language
  - one confusion-matrix heatmap per language (with enough samples)

Usage:
    python analysis/plot_language_metrics.py outputs/gemma_language_metrics.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless — no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np

MIN_SAMPLES_FOR_CONFUSION_PLOT = 20


def plot_accuracy_bar(results: dict, out_dir: str, model_name: str) -> None:
    per_lang = results["per_language"]
    langs, accs, f1s = [], [], []
    for lang, block in per_lang.items():
        if block.get("num_labeled", 0) == 0:
            continue
        langs.append(lang)
        accs.append(block.get("accuracy", 0))
        f1s.append(block.get("macro_f1", 0))

    if not langs:
        print("[plot] no languages with labeled data to plot.")
        return

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.2), 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-language accuracy and macro F1 — {model_name}")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{model_name}_per_language_accuracy_f1.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote -> {out_path}")


def plot_confusion_matrices(results: dict, out_dir: str, model_name: str) -> None:
    per_lang = results["per_language"]
    for lang, block in per_lang.items():
        cm_block = block.get("confusion_matrix")
        if not cm_block or block.get("num_labeled", 0) < MIN_SAMPLES_FOR_CONFUSION_PLOT:
            continue

        labels = cm_block["labels"]
        matrix = np.array(cm_block["matrix"])

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(f"{lang}\n({model_name})", fontsize=10)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        safe_lang_name = "".join(c if c.isalnum() else "_" for c in lang)
        out_path = os.path.join(out_dir, f"{model_name}_confusion_{safe_lang_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/plot_language_metrics.py outputs/<model>_language_metrics.json")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    model_name = os.path.basename(input_path).replace("_language_metrics.json", "")
    out_dir = os.path.join("outputs", "plots", model_name)
    os.makedirs(out_dir, exist_ok=True)

    plot_accuracy_bar(results, out_dir, model_name)
    plot_confusion_matrices(results, out_dir, model_name)
