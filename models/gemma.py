import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class GemmaModel:

    def __init__(self, model_name):

        print(f"\nLoading {model_name}...\n")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(

            model_name,

            torch_dtype=torch.float16,

            device_map="auto"

        )

    def generate(self, prompt, generation_config):

    inputs = self.tokenizer(
        prompt,
        return_tensors="pt"
    ).to(self.model.device)

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=generation_config["max_new_tokens"],
        do_sample=generation_config["do_sample"],
        temperature=generation_config["temperature"]
    )

    # Decode only the generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = self.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()
