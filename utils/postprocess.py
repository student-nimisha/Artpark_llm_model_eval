"""
Normalizes a raw model completion (or a dataset's gold label, which may
be an int, bool, or string) into "safe", "not safe", or "unknown".
"""


def normalize_label(raw: object) -> str:
    text = str(raw).strip().lower()

    if text in ("1", "true", "profane", "not_safe", "notsafe"):
        return "not safe"
    if text in ("0", "false", "not_profane", "notprofane"):
        return "safe"

    if "not safe" in text or "unsafe" in text or "not-safe" in text or "profane" in text:
        return "not safe"
    if "safe" in text:
        return "safe"

    return "unknown"
