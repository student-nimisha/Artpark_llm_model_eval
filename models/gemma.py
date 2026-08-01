import torch

from transformers import pipeline


class GemmaModel:

    def __init__(self, model_name):

        print(f"\nLoading {model_name}...\n")

        self.pipe = pipeline(
            task="text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt, generation_config):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        output = self.pipe(
            messages,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config["do_sample"]
        )

        # Extract only the assistant's reply
        answer = output[0]["generated_text"][-1]["content"]

        return answer.strip()
