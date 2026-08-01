import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class GemmaModel:

    def __init__(self, model_name):

        print(f"\nLoading {model_name}...\n")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, generation_config):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                inputs,
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"]
            )

        generated_tokens = outputs[0][inputs.shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()
