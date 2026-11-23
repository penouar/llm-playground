from .providers import FakeProvider


def main() -> None:
    provider = FakeProvider()
    full_text = ""
    for chunk in provider.stream([]):
        print(chunk, end="", flush=True)
        full_text += chunk
    print()
    print("Full text:", full_text)
    