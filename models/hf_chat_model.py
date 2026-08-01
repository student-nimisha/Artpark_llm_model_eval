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
     through text.

THE FIX
-------
Work at the TOKEN level:
  1. tokenizer.apply_chat_template(messages, tokenize=True,
     add_generation_prompt=True, return_dict=True, return_tensors="pt")
     -> a dict with "input_ids" and "attention_mask".
     NOTE: we explicitly pass return_dict=True. Some chat models (Gemma-3
     among them) always return a dict/BatchEncoding from
     apply_chat_template regardless of this flag, since their tokenizer
     is really a multimodal-capable processor under the hood. Requesting
     the dict explicitly makes the code correct and uniform across every
     model family instead of assuming a plain tensor comes back.
  2. Record input_ids.shape[-1] = prompt length in tokens.
  3. model.generate(**encoded, ...) — passing attention_mask too, not
     just input_ids.
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
            dtype=dtype,  # `torch_dtype` kwarg is deprecated as of recent transformers
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Several chat models (Gemma among them) ship without a pad token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_inputs(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if not self.supports_system_role:
            messages = self._merge_system_into_user(messages)

        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # encoded is a dict/BatchEncoding: {"input_ids": ..., "attention_mask": ...}
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        return encoded

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
        encoded = self._build_inputs(messages)
        prompt_len = encoded["input_ids"].shape[-1]

        gen_kwargs = dict(generation_config)
        # Setting both max_length and max_new_tokens triggers a warning and
        # ambiguous behaviour — we only ever want the latter.
        gen_kwargs.pop("max_length", None)
        gen_kwargs.setdefault("max_new_tokens", 8)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        output_ids = self.model.generate(**encoded, **gen_kwargs)

        # Token-level slice, NOT string-level — this is the actual fix
        # for empty predictions.
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
