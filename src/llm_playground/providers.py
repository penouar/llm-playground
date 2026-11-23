from abc import ABC, abstractmethod
from collections.abc import Iterator

from openai import OpenAI


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Send messages to the model and yield the answer in text chunks."""


class FakeProvider(LLMProvider):
    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield "Hello"
        yield " from"
        yield " the fake provider"


class OllamaProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content