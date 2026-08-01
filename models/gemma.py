"""
Gemma-3 instruct family (e.g. google/gemma-3-4b-it).

Everything generic lives in HFChatModel. The only Gemma-specific fact is
that its chat template does not accept a `system` role message.
"""

from models.hf_chat_model import HFChatModel


class GemmaModel(HFChatModel):
    supports_system_role = False
