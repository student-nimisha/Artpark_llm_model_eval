"""
Streams a Hugging Face dataset and normalizes each example to
{"text": ..., "label": ...}, regardless of what the source dataset calls
its columns.

I couldn't confirm the exact column names of
mangalathkedar/multilingual-indic-profane, so this auto-detects from
common candidates and lets the config override explicitly if it guesses
wrong. Watch the printed log line on first run to confirm.
"""

from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset

TEXT_COLUMN_CANDIDATES = ("text", "transcript", "sentence", "content", "comment", "utterance")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "is_profane", "profane", "class", "target", "profanity")


def _detect_columns(dataset, ds_cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    text_col = ds_cfg.get("text_column")
    label_col = ds_cfg.get("label_column")

    if text_col and label_col:
        return text_col, label_col

    # Peek at the first example to discover column names. Works for both
    # streaming (IterableDataset) and in-memory datasets.
    first = next(iter(dataset))

    if text_col is None:
        text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in first), None)
    if label_col is None:
        label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in first), None)

    if text_col is None:
        raise ValueError(
            "Could not auto-detect a text column. "
            f"Available columns: {list(first.keys())}. "
            "Set dataset.text_column explicitly in the config."
        )

    return text_col, label_col


def load_streaming_dataset(config: Dict[str, Any]):
    ds_cfg = config["dataset"]

    dataset = load_dataset(
        ds_cfg["name"],
        split=ds_cfg.get("split", "train"),
        streaming=ds_cfg.get("streaming", True),
    )

    text_col, label_col = _detect_columns(dataset, ds_cfg)
    print(f"[hf_dataset_loader] using text_column='{text_col}', label_column='{label_col}'")

    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": example.get(text_col, ""),
            "label": example.get(label_col) if label_col else None,
        }

    return dataset.map(_preprocess)
