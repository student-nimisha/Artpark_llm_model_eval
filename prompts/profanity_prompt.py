"""
Returns a chat-format `messages` list rather than a raw formatted string.
This keeps prompt building model-agnostic: chat-template formatting is
the model wrapper's job (models/hf_chat_model.py), not the prompt
builder's. The same messages list feeds Gemma, Aya, Qwen, or Llama
unchanged.
"""

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are an expert multilingual profanity detection system. "
    "You judge text written in any language, including Indic languages "
    "and code-mixed text."
)


def build_prompt(text: str) -> List[Dict[str, str]]:
    user_content = (
        "Classify the following text.\n"
        "Return ONLY one of these two labels and nothing else:\n"
        "safe\n"
        "not safe\n\n"
        f"Text: {text}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
