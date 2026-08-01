def build_prompt(text):

    return f"""
You are an expert multilingual profanity detection system.

Classify the text below.

Return ONLY one of these labels:

safe

or

not safe

Text:
{text}

Answer:
""".strip()
