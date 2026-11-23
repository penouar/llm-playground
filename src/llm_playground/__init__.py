import os

from dotenv import load_dotenv

from .providers import OllamaProvider


def main() -> None:
    load_dotenv()
    provider = OllamaProvider(os.environ["OLLAMA_MODEL"])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a token is in an LLM, in about 10 sentences."},
    ]
    full_text = ""
    for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
        full_text += chunk
    print()