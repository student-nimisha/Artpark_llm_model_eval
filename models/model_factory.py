"""
Selects a model wrapper class purely from config["model"]["name"].

To add Aya Expanse / Qwen / Llama / Sarvam:
  1. If it needs no special handling beyond the generic chat pattern
     (most modern instruct models), do nothing — it already works via
     the HFChatModel fallback below.
  2. If it needs a quirk (no system role, custom stop tokens, etc.), add
     a small subclass in models/<name>.py the same way models/gemma.py
     does, then add one line to _REGISTRY.

evaluator/evaluators.py and main.py never need to change either way.
"""

from typing import Any, Dict

from models.base_model import BaseModel
from models.gemma import GemmaModel
from models.hf_chat_model import HFChatModel

# Substring-matched against the lowercased model name. Order doesn't
# matter; first match wins.
_REGISTRY = {
    "gemma": GemmaModel,
    # "aya": AyaModel,        # models/aya.py, once added
    # "qwen": QwenModel,      # models/qwen.py, once added
    # "llama": LlamaModel,    # models/llama.py, once added
    # "sarvam": SarvamModel,  # models/sarvam.py, once added
}


def load_model(config: Dict[str, Any]) -> BaseModel:
    model_name = config["model"]["name"].lower()

    model_cls = HFChatModel  # safe generic default for unlisted models
    for key, cls in _REGISTRY.items():
        if key in model_name:
            model_cls = cls
            break

    model = model_cls(config)
    model.load()
    return model
