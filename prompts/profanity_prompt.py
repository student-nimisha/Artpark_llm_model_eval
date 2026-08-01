
def build_prompt(text):

    return f"""
You are an expert multilingual profanity detection system.

Classify the following text.

Return EXACTLY one word.

safe
or
not safe

Do not explain.
Do not give a reason.
Do not output anything else.

Text:
{text}

Answer:
""".strip()
