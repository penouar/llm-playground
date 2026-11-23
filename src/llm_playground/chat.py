import openai

from .providers import LLMProvider

SYSTEM_PROMPT = "You are a helpful assistant."


def run_chat(provider: LLMProvider) -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Type your message. Type 'exit' to stop.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() == "exit":
            break
        if not user_text:
            continue
        messages.append({"role": "user", "content": user_text})
        print("Assistant: ", end="", flush=True)
        full_text = ""
        try:
            for chunk in provider.stream(messages):
                print(chunk, end="", flush=True)
                full_text += chunk
        except openai.APIError as error:
            print(f"\n[Provider error] {error}")
            print(getattr(error, "body", None))
            messages.pop()