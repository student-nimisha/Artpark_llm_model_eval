"""
Abstract interface every model wrapper must implement.

The evaluator only ever calls:
    model.generate(messages, generation_config) -> str
So any new model wrapper that honours this contract works without any
change to evaluator/evaluators.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model"]["name"]
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        """Load tokenizer + model weights onto the correct device."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        """
        messages: chat-format list, e.g. [{"role": "user", "content": "..."}]
        Returns: ONLY the newly generated completion text (prompt stripped,
        special tokens stripped). Never the echoed prompt.
        """
        raise NotImplementedError
