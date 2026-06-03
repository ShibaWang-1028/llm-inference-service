"""Request schema validation tests."""

import pytest
from pydantic import ValidationError

from app.schemas import ChatCompletionRequest


def test_valid_request_parses() -> None:
    req = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert req.model == "m"
    assert req.stream is False


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"model": "m", "messages": []})


def test_missing_model_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})


def test_extra_fields_allowed() -> None:
    # OpenAI clients send fields we don't model (e.g. presence_penalty); keep them.
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "presence_penalty": 0.5,
            "seed": 42,
        }
    )
    dumped = req.model_dump(exclude_unset=True)
    assert dumped["presence_penalty"] == 0.5
    assert dumped["seed"] == 42


def test_multimodal_content_allowed() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "describe this"}]}],
        }
    )
    assert isinstance(req.messages[0].content, list)
