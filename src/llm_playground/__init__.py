import argparse
import os

from dotenv import load_dotenv

from .chat import run_chat
from .providers import (
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenRouterProvider,
)


def build_provider(name: str) -> LLMProvider:
    if name == "ollama":
        return OllamaProvider(os.environ["OLLAMA_MODEL"])
    if name == "gemini":
        return GeminiProvider(
            os.environ["GEMINI_MODEL"],
            os.environ["GEMINI_API_KEY"],
        )
    return OpenRouterProvider(
        os.environ["OPENROUTER_MODEL"],
        os.environ["OPENROUTER_API_KEY"],
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLI chat with swappable LLM providers")
    parser.add_argument(
        "--provider", choices=["ollama", "openrouter", "gemini"], default="ollama"
    )
    parser.add_argument("--json", action="store_true", help="ask for a validated JSON reply")
    args = parser.parse_args()
    run_chat(build_provider(args.provider), json_mode=args.json)