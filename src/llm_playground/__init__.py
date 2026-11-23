import os

from dotenv import load_dotenv

from .chat import run_chat
from .providers import OllamaProvider


def main() -> None:
    load_dotenv()
    provider = OllamaProvider(os.environ["OLLAMA_MODEL"])
    run_chat(provider)