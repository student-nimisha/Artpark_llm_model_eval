"""
Generic wrapper for any modern instruction-tuned causal LM that ships a chat
template: Gemma, Llama-3.x, Qwen2.x, Aya-Expanse, Mistral-Instruct, etc.

WHY NOT transformers.pipeline("text-generation")?
--------------------------------------------------
`pipeline` in chat mode hides two things a generic evaluator needs explicit
control over:
  1. Not every chat template accepts a `system` role (Gemma's doesn't;
     Llama/Qwen/Aya's do).
  2. Exactly where the prompt ends and the completion begins — pipeline
     hands back the full re-rendered conversation or a raw string, and
     slicing that STRING against the original prompt is fragile.

THE FIX
-------
Work at the TOKEN level:
  1. tokenizer.apply_chat_template(..., return_dict=True, tokenize=True,
     return_tensors="pt") -> dict with "input_ids" + "attention_mask".
  2. Record input_ids.shape[-1] = prompt length in tokens.
  3. model.generate(**encoded, ...)
  4. Slice output_ids[0][prompt_len:] and decode only the new tokens.

OPTIONAL 4-BIT QUANTIZATION
----------------------------
Large models (e.g. Gemma-3 27B, Aya Expanse 32B) may not fit in bf16 on a
Kaggle T4 x2. Setting "load_in_4bit": true in a config's "model" block
loads the model in 4-bit via bitsandbytes instead — same interface, same
generate() call, just less VRAM.

PER-SAMPLE GENERATION TIMEOUT
-------------------------------
A single pathological input (very long text, unusual characters) can
occasionally cause generate() to run far longer than normal, effectively
hanging the whole evaluation loop. Setting "generation_timeout_seconds"
in the top-level config (default 60) forces any single generate() call to
give up and return an empty string after that many seconds, logging a
warning, instead of hanging indefinitely. Combined with the evaluator's
checkpoint/resume logic, this means one bad sample can never stall an
entire multi-hour run.
"""

import signal
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.base_model import BaseModel


class GenerationTimeout(Exception):
    pass


class HFChatModel(BaseModel):
    # Override to False in a subclass if the model's chat template raises
    # on a "system" message (Gemma is the current example).
    supports_system_role: bool = True

    def load(self) -> None:
        model_cfg = self.config["model"]
        dtype_str = model_cfg.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        trust_remote_code = model_cfg.get("trust_remote_code", False)
        load_in_4bit = model_cfg.get("load_in_4bit", False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=None if load_in_4bit else dtype,
            quantization_config=quantization_config,
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_timeout_seconds = self.config.get("generation_timeout_seconds", 60)

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

    def _generate_with_timeout(self, encoded: Dict[str, torch.Tensor], gen_kwargs: Dict[str, Any]):
        def _handler(signum, frame):
            raise GenerationTimeout()

        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.generation_timeout_seconds)

        try:
            return self.model.generate(**encoded, **gen_kwargs)
        except GenerationTimeout:
            print(
                f"[hf_chat_model] WARNING: generation exceeded "
                f"{self.generation_timeout_seconds}s — skipping this sample."
            )
            return None
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        encoded = self._build_inputs(messages)
        prompt_len = encoded["input_ids"].shape[-1]

        gen_kwargs = dict(generation_config)
        gen_kwargs.pop("max_length", None)
        gen_kwargs.setdefault("max_new_tokens", 8)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        output_ids = self._generate_with_timeout(encoded, gen_kwargs)
        if output_ids is None:
            return ""

        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
