import openai
from pydantic import ValidationError

from .providers import LLMProvider
from .schema import Reply, clean_json

SYSTEM_PROMPT = "You are a helpful assistant."
JSON_SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply ONLY with a JSON object, no other text. "
    'Shape: {"answer": <text>, "confidence": <integer from 0 to 100>}. '
    "The confidence is a plain number, never text and never with a % sign. "
    'Example: {"answer": "Paris is the capital of France.", "confidence": 95}'
)


def print_usage(provider: LLMProvider) -> None:
    usage = provider.last_usage
    if usage is None:
        print("[tokens: not reported by this provider]")
    else:
        print(
            f"[tokens: in {usage.input_tokens} | "
            f"out {usage.output_tokens} | total {usage.total}]"
        )


def stream_text(provider: LLMProvider, messages: list[dict]) -> str:
    """Stream the answer to the screen and return the full joined text."""
    full_text = ""
    for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
        full_text += chunk
    print()
    print_usage(provider)
    return full_text


def describe(error: ValidationError) -> str:
    """A short, readable version of a Pydantic error."""
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'reply'}: {e['msg']}"
        for e in error.errors()
    )


def get_json_reply(provider: LLMProvider, messages: list[dict]) -> Reply:
    """Get a valid Reply. If the first answer is invalid, retry exactly once."""
    text = stream_text(provider, messages)
    try:
        reply = Reply.model_validate_json(clean_json(text))
    except ValidationError as error:
        print(f"[Invalid JSON, retrying once] {describe(error)}")
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                f"Your last reply was invalid: {describe(error)}. "
                "Reply again with ONLY valid JSON in the required shape."
            ),
        })
        print("Assistant (retry): ", end="", flush=True)
        text = stream_text(provider, messages)
        # No try here: a second failure goes up to the caller. That is "only once".
        reply = Reply.model_validate_json(clean_json(text))
    messages.append({"role": "assistant", "content": text})
    return reply


def run_chat(provider: LLMProvider, json_mode: bool = False) -> None:
    system_prompt = JSON_SYSTEM_PROMPT if json_mode else SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    print("Type your message. Type 'exit' to stop.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() == "exit":
            break
        if not user_text:
            continue
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_text})
        print("Assistant: ", end="", flush=True)
        try:
            if json_mode:
                reply = get_json_reply(provider, messages)
                print(f"Parsed -> answer: {reply.answer} | confidence: {reply.confidence}")
            else:
                text = stream_text(provider, messages)
                messages.append({"role": "assistant", "content": text})
        except openai.APIError as error:
            print(f"\n[Provider error] {error}")
            del messages[checkpoint:]
        except ValidationError as error:
            print(f"[Could not get valid JSON after one retry] {describe(error)}")
            del messages[checkpoint:]