"""
Qwen3 instruct family — using Qwen/Qwen3-4B-Instruct-2507 specifically.

WHY THE "-2507" CHECKPOINT AND NOT PLAIN Qwen3-4B
-----------------------------------------------------
Qwen3's dense models support a "thinking" mode by default: before
answering, the model can emit a <think>...</think> reasoning block. That
is valuable for complex reasoning tasks, but actively harmful here — our
task only needs a short "safe" / "not safe" label within a handful of
tokens, and thinking tokens would consume the entire generation budget
before the model ever produces the actual label. Qwen released the
"-2507" dated checkpoints specifically as non-thinking-by-default
variants of the same model, which is the correct choice for short,
single-label classification tasks like this one.

Qwen's chat template natively supports a system role (like Aya), so no
role-merging override is needed.
"""

from models.hf_chat_model import HFChatModel


class QwenModel(HFChatModel):
    supports_system_role = True
