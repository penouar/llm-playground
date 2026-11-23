import pytest
from pydantic import ValidationError

from llm_playground.chat import get_json_reply
from llm_playground.providers import ScriptedProvider

BAD = '{"answer": "Hi", "confidence": "90%"}'
GOOD = '{"answer": "Hi", "confidence": 90}'


def make_messages() -> list[dict]:
    return [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]


def test_good_json_needs_no_retry():
    provider = ScriptedProvider([GOOD])
    reply = get_json_reply(provider, make_messages())
    assert reply.confidence == 90
    assert provider.calls == 1


def test_bad_json_is_retried_once_then_succeeds():
    provider = ScriptedProvider([BAD, GOOD])
    reply = get_json_reply(provider, make_messages())
    assert reply.confidence == 90
    assert provider.calls == 2


def test_bad_json_twice_stops_after_one_retry():
    provider = ScriptedProvider([BAD, BAD, GOOD])
    with pytest.raises(ValidationError):
        get_json_reply(provider, make_messages())
    assert provider.calls == 2  # the third reply was never requested


def test_code_fences_are_removed():
    fenced = "```json\n" + GOOD + "\n```"
    provider = ScriptedProvider([fenced])
    reply = get_json_reply(provider, make_messages())
    assert reply.answer == "Hi"