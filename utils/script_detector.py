"""
Detects the likely script (as a proxy for language family) of a short
text sample using Unicode code-point ranges.

WHY SCRIPT DETECTION INSTEAD OF A STATISTICAL LANGUAGE-ID MODEL
-------------------------------------------------------------------
Statistical language-ID tools (langdetect, fastText langid) are trained
on longer, cleaner paragraphs and are unreliable on short, noisy inputs
like single sentences or single words, which is what this dataset
contains. Checking which Unicode block a character belongs to is
deterministic, fast, and accurate even on very short text, entirely
offline, with no model to download or run.

LIMITATION (stated honestly)
--------------------------------
Script is not a perfect proxy for language: Devanagari is shared by
Hindi, Marathi, Nepali, and Sanskrit; Bengali script is shared by Bengali
and Assamese. Those buckets are labeled by script name, not asserted as
a single language. Dravidian scripts (Tamil, Telugu, Kannada, Malayalam)
are each unique to one language, so those buckets are fully accurate.
"""

from collections import Counter

SCRIPT_RANGES = {
    "Devanagari (Hindi/Marathi/Nepali/Sanskrit)": [(0x0900, 0x097F)],
    "Bengali/Assamese": [(0x0980, 0x09FF)],
    "Gurmukhi (Punjabi)": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Sinhala": [(0x0D80, 0x0DFF)],
    "Urdu/Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Latin (English/Romanized)": [(0x0041, 0x005A), (0x0061, 0x007A)],
}


def detect_script(text: str) -> str:
    """Returns the name of the script with the most matching characters
    in `text`. Falls back to "unknown" if nothing recognizable is found
    (e.g. text is only punctuation/digits/emoji, or the text is corrupted)."""
    if not text:
        return "unknown"

    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] += 1
                break

    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]
