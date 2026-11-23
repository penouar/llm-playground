from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    last_usage: Usage | None = None

    @abstractmethod
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Send messages to the model and yield the answer in text chunks.

        After the stream ends, last_usage holds the token counts (or None).
        """


class FakeProvider(LLMProvider):
    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield "Hello"
        yield " from"
        yield " the fake provider"


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


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        self.last_usage = None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            if chunk.usage:
                self.last_usage = Usage(
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens,
                )
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self, model: str) -> None:
        super().__init__(model, "http://localhost:11434/v1", "ollama")


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, "https://openrouter.ai/api/v1", api_key)