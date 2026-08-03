"""
Phase 2, Step 1 of the two-phase evaluation pipeline.

Takes an EXISTING predictions CSV (already produced by a completed model
run — no LLM inference happens here) and adds a "language" column,
inferred from the "text" column via Unicode script detection.

Never modifies the original file. Writes a new file instead:
    outputs/<model>_predictions.csv
        -> outputs/<model>_predictions_with_language.csv

Usage:
    python analysis/add_language.py outputs/gemma_predictions.csv
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.script_detector import detect_script

UNKNOWN_WARNING_THRESHOLD = 0.10  # warn if >10% of rows can't be script-identified


def add_language_column(input_csv: str, output_csv: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Could not find {input_csv}. Make sure the model run that "
            "produces this file has actually completed."
        )

    # utf-8-sig safely handles a BOM if the file was ever touched by Excel;
    # falls through fine for plain utf-8 files too.
    with open(input_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None or "text" not in fieldnames:
        raise ValueError(
            f"'{input_csv}' has no 'text' column (found: {fieldnames}). "
            "Cannot perform language detection."
        )

    language_counts = Counter()
    for row in rows:
        lang = detect_script(row["text"])
        row["language"] = lang
        language_counts[lang] += 1

    total = len(rows)
    unknown_ratio = language_counts.get("unknown", 0) / total if total else 0.0

    out_fieldnames = fieldnames + ["language"]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[add_language] read {total} rows from {input_csv}")
    print("[add_language] language distribution:")
    for lang, count in language_counts.most_common():
        print(f"  {lang:45s} {count:6d}  ({100 * count / total:.1f}%)")
    print(f"[add_language] wrote -> {output_csv}")

    if unknown_ratio > UNKNOWN_WARNING_THRESHOLD:
        print(
            f"\n[add_language] WARNING: {100 * unknown_ratio:.1f}% of rows were "
            "classified as 'unknown' script. This can indicate an ENCODING "
            "PROBLEM rather than genuinely unrecognizable text (e.g. Malayalam "
            "text got corrupted/mojibake'd when the CSV was written or "
            "re-saved, most commonly by opening it in Excel). Recommended checks:\n"
            "  1. Open a few 'unknown' rows and inspect the raw text field directly.\n"
            "  2. If it looks like garbled characters (e.g. repeated '?' or\n"
            "     '\\xXX' byte sequences) instead of actual script characters,\n"
            "     the file was saved/re-saved with the wrong encoding at some point.\n"
            "  3. Always read/write these CSVs with encoding='utf-8' — never\n"
            "     open and re-save them in Excel, which silently mangles\n"
            "     non-Latin text.\n"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/add_language.py outputs/<model>_predictions.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_with_language{ext}"

    add_language_column(input_path, output_path)
