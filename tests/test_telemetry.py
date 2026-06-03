"""Telemetry tests: the Langfuse tracker calls the right v4 SDK method."""

from unittest.mock import MagicMock

from app.config import Settings
from app.telemetry import LangfuseTracker


def test_tracker_disabled_is_noop() -> None:
    t = LangfuseTracker(Settings(_env_file=None, enable_langfuse=False))
    assert t.enabled is False
    # should not raise even though there's no client
    t.log_chat(model="m", messages=[], output="x", usage=None, latency_ms=1.0)


def test_log_chat_uses_start_observation() -> None:
    t = LangfuseTracker(Settings(_env_file=None, enable_langfuse=False))
    # inject a fake client and force-enable, to assert the call shape
    t._client = MagicMock()
    t.enabled = True
    t.log_chat(
        model="Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "hi"}],
        output="hello",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        latency_ms=12.0,
    )
    assert t._client.start_observation.called
    kwargs = t._client.start_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["model"] == "Qwen2.5-7B-Instruct"
    assert kwargs["usage_details"] == {"input": 1, "output": 2, "total": 3}
    t._client.start_observation.return_value.end.assert_called_once()


def test_log_chat_swallows_errors() -> None:
    t = LangfuseTracker(Settings(_env_file=None, enable_langfuse=False))
    t._client = MagicMock()
    t._client.start_observation.side_effect = RuntimeError("boom")
    t.enabled = True
    t.log_chat(model="m", messages=[], output=None, usage=None, latency_ms=1.0)
    # a failure disables the tracker instead of propagating
    assert t.enabled is False
