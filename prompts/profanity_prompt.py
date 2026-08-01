
def build_prompt(text):

    prompt = f"""
You are an expert multilingual profanity detection system.

Your task is to determine whether the given text contains profanity.

Return ONLY one of these labels.

safe
not safe

Text:

{text}

Answer:
"""

    return prompt.strip()
