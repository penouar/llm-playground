# 🤖 llm-playground

> A small command-line chat app that talks to **different LLM providers through one common interface**, with streaming, validated JSON replies, and token tracking.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Providers](https://img.shields.io/badge/providers-Ollama%20%7C%20OpenRouter%20%7C%20Gemini-purple)

I built this to learn the building blocks behind every LLM app: chat roles, streaming, JSON output, tokens, and clean provider abstraction.

---

## ✨ Features

| | Feature | What it does |
|---|---|---|
| 💬 | **Chat with memory** | Keeps the full message list and resends it every turn |
| ⚡ | **Streaming** | Prints the answer chunk by chunk as the model writes it |
| 🔌 | **Swappable providers** | One `LLMProvider` interface, three implementations |
| 🧾 | **Validated JSON mode** | `--json` checks the reply with Pydantic |
| 🔁 | **Retry once** | Invalid JSON is sent back to the model with the error, one time only |
| 🔢 | **Token usage** | Shows input, output, and total tokens after each answer |
| 🧪 | **Tests** | The retry logic is tested with a scripted fake provider |

---

## 🧱 How it works

```
            CLI chat loop  (chat.py)
                   │
        LLMProvider interface
          stream(messages)
                   │
     ┌─────────────┼──────────────┐
     │             │              │
  Ollama      OpenRouter       Gemini
  (local)      (hosted)       (hosted)
```

1. The chat loop keeps a `messages` list (`system`, `user`, `assistant`).
2. It calls `provider.stream(messages)` and prints each chunk as it arrives.
3. In `--json` mode, it joins the chunks, removes code fences, and validates the text with Pydantic.
4. If validation fails, the bad reply and the error go back to the model **once**. A second failure stops with a clear message.
5. Token usage is read from the provider after the stream ends.

---

## 🚀 Quick start

### 1. Requirements
- 🐍 [Python 3.12](https://www.python.org/)
- ⚙️ [uv](https://docs.astral.sh/uv/)
- 🦙 [Ollama](https://ollama.com) (for the free local model)

### 2. Install
```bash
git clone https://github.com/YOUR-USERNAME/llm-playground.git
cd llm-playground
uv sync
ollama pull llama3.2:1b
```

### 3. Configure
Copy `.env.example` to `.env` and fill in only what you need:
```env
OLLAMA_MODEL=llama3.2:1b
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=model-id-here
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=model-id-here
```
> 🔒 `.env` is in `.gitignore`. Never commit your keys.

---

## 🎮 Usage

```bash
# Local model, normal chat
uv run llm-playground --provider ollama

# Local model, validated JSON replies
uv run llm-playground --provider ollama --json

# Hosted providers
uv run llm-playground --provider gemini --json
uv run llm-playground --provider openrouter
```

Type `exit` to stop.

### Example (`--json` mode)
```text
You: what is a token
Assistant: {"answer": "A token is a small piece of text...", "confidence": 90}
[tokens: in 62 | out 38 | total 100]
Parsed -> answer: A token is a small piece of text... | confidence: 90
```

### JSON reply format
```json
{ "answer": "text", "confidence": 0-100 }
```

---

## 📁 Project structure

```
llm-playground/
├── src/llm_playground/
│   ├── __init__.py     # CLI entry point and provider selection
│   ├── chat.py         # chat loop, streaming, JSON retry
│   ├── providers.py    # LLMProvider interface + providers
│   └── schema.py       # Pydantic model and JSON cleaning
├── tests/
│   └── test_retry.py   # retry tests with a scripted fake provider
├── .env.example        # template for your settings
├── pyproject.toml
└── README.md
```

---

## 🧪 Tests

```bash
uv run pytest -v
```

The tests use a `ScriptedProvider` that returns prepared answers, so the retry behavior is proven on demand:
- ✅ Good JSON needs no retry
- 🔁 Bad JSON is retried once, then succeeds
- 🛑 Bad JSON twice stops after one retry
- 🧹 Code fences around JSON are removed

---

## 📌 Status

- ✅ Fully tested with **Ollama** (streaming, memory, JSON mode, token usage)
- 🔌 **OpenRouter** and **Gemini** are implemented with the same interface. The requests reach the services, but free-tier capacity returned `429` and `503` errors during my tests.

---

## 🧠 What I learned

- The model has **no memory**. The app resends the whole message list each turn, so input tokens grow with every message.
- Streaming chunks are **not valid JSON on their own**. Join them first, then validate.
- Pydantic checks the **shape and types**, not whether the answer is true.
- Small local models break JSON more often. A clear example in the prompt helps.
- A provider interface makes switching models easy, and testing easy (fake providers).


---

## 🗺️ Roadmap
- [ ] Automatic retry with backoff for `429` and `503` provider errors
- [ ] Save and load conversations
- [ ] Show cost estimates from token usage
