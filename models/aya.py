import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class AyaModel:

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

            temperature=generation_config["temperature"],

            do_sample=generation_config["do_sample"]

        )

        answer = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        return answer
