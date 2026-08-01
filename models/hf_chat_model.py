"""
Generic wrapper for any modern instruction-tuned causal LM that ships a chat
template: Gemma, Llama-3.x, Qwen2.x, Aya-Expanse, Mistral-Instruct, etc.

WHY NOT transformers.pipeline("text-generation")?
--------------------------------------------------
`pipeline` in chat mode hides two things a generic evaluator needs explicit
control over:
  1. Not every chat template accepts a `system` role (Gemma's doesn't;
     Llama/Qwen's do).
  2. Exactly where the prompt ends and the completion begins. Pipeline
     hands back either the full re-rendered conversation or a raw string,
     and slicing that STRING against the original prompt is fragile —
     special tokens and chat-template whitespace don't round-trip cleanly
     through text. This is almost certainly why `prediction` came back
     empty: string-diffing found nothing beyond the prompt to return.

THE FIX
-------
Work at the TOKEN level:
  1. tokenizer.apply_chat_template(messages, tokenize=True,
     add_generation_prompt=True, return_tensors="pt") -> input_ids
  2. Record input_ids.shape[-1] = prompt length in tokens.
  3. model.generate(input_ids, ...)
  4. Slice output_ids[0][prompt_len:] (tokens, not characters) and decode
     only that slice with skip_special_tokens=True.
"""

from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base_model import BaseModel


class HFChatModel(BaseModel):
    # Override to False in a subclass if the model's chat template raises
    # on a "system" message (Gemma is the current example).
    supports_system_role: bool = True

    def load(self) -> None:
        model_cfg = self.config["model"]
        dtype_str = model_cfg.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        trust_remote_code = model_cfg.get("trust_remote_code", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Several chat models (Gemma among them) ship without a pad token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_input_ids(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        if not self.supports_system_role:
            messages = self._merge_system_into_user(messages)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.model.device)

        return input_ids

    @staticmethod
    def _merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fold any system message into the first user turn, for templates
        (Gemma) that only understand user/assistant/model roles."""
        system_chunks = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]

        if not system_chunks or not rest:
            return rest or messages

        if rest[0]["role"] == "user":
            merged_content = "\n\n".join(system_chunks + [rest[0]["content"]])
            rest[0] = {"role": "user", "content": merged_content}

        return rest

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        input_ids = self._build_input_ids(messages)
        prompt_len = input_ids.shape[-1]

        gen_kwargs = dict(generation_config)
        # Setting both max_length and max_new_tokens triggers a warning and
        # ambiguous behaviour — we only ever want the latter.
        gen_kwargs.pop("max_length", None)
        gen_kwargs.setdefault("max_new_tokens", 8)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        output_ids = self.model.generate(input_ids, **gen_kwargs)

        # Token-level slice, NOT string-level — this is the actual fix.
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
