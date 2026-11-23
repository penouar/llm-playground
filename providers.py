from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Send messages to the model and yield the answer in text chunks."""


class FakeProvider(LLMProvider):
    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield "Hello"
        yield " from"
        yield " the fake provider"