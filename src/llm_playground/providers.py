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


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model, "http://localhost:11434/v1", "ollama")


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, "https://openrouter.ai/api/v1", api_key)

class ScriptedProvider(LLMProvider):
    """Fake provider that returns prepared answers, one per call. For tests."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def stream(self, messages: list[dict]) -> Iterator[str]:
        reply = self.replies[self.calls]
        self.calls += 1
        for i in range(0, len(reply), 5):
            yield reply[i : i + 5]