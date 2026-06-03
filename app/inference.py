"""Forwarding logic between the gateway and the local vLLM OpenAI server.

We run vLLM's own OpenAI-compatible server on localhost and proxy to it. This
keeps us on vLLM's stable HTTP contract instead of its fast-moving Python
internals, and it lets the whole gateway run against a fake upstream with no GPU.

For streaming we pass the upstream's SSE bytes straight through (exact fidelity)
while parsing a decoded copy on the side to capture time-to-first-token and the
final usage block for metrics and cost tracking.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.telemetry import INFLIGHT, LATENCY, REQUESTS, TOKENS, TTFT, LangfuseTracker

logger = logging.getLogger("gateway.inference")

CHAT_PATH = "/v1/chat/completions"
ROUTE = "chat.completions"


def build_upstream_client(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    # Generous read timeout for long generations; keep connect short.
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    return httpx.AsyncClient(base_url=base_url, timeout=timeout)


async def is_upstream_ready(client: httpx.AsyncClient) -> bool:
    """Used by /health/ready. vLLM serves /health once the model is loaded."""
    try:
        r = await client.get("/health", timeout=5.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _record(status: int, elapsed: float, usage: dict[str, Any] | None) -> None:
    REQUESTS.labels(ROUTE, "POST", str(status)).inc()
    LATENCY.labels(ROUTE).observe(elapsed)
    if usage:
        TOKENS.labels("prompt").inc(usage.get("prompt_tokens", 0))
        TOKENS.labels("completion").inc(usage.get("completion_tokens", 0))


def _log_langfuse(
    tracker: LangfuseTracker,
    payload: dict[str, Any],
    data: dict[str, Any] | None,
    usage: dict[str, Any] | None,
    elapsed: float,
) -> None:
    output = None
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            output = msg.get("content")
    tracker.log_chat(
        model=payload.get("model"),
        messages=payload.get("messages"),
        output=output,
        usage=usage,
        latency_ms=round(elapsed * 1000, 1),
    )


async def complete_chat(
    client: httpx.AsyncClient, payload: dict[str, Any], tracker: LangfuseTracker
) -> tuple[int, Any]:
    """Non-streaming completion. Returns (status_code, json_body)."""
    t0 = time.perf_counter()
    INFLIGHT.inc()
    try:
        r = await client.post(CHAT_PATH, json=payload)
        elapsed = time.perf_counter() - t0
        try:
            data = r.json()
        except json.JSONDecodeError:
            data = {"error": {"message": "upstream returned a non-JSON response"}}
        usage = data.get("usage") if isinstance(data, dict) else None
        _record(r.status_code, elapsed, usage)
        if r.status_code == 200:
            _log_langfuse(tracker, payload, data, usage, elapsed)
        return r.status_code, data
    finally:
        INFLIGHT.dec()


async def stream_chat(
    client: httpx.AsyncClient, payload: dict[str, Any], tracker: LangfuseTracker
) -> AsyncIterator[bytes]:
    """Streaming completion. Yields raw SSE bytes from the upstream."""
    t0 = time.perf_counter()
    INFLIGHT.inc()
    first = True
    usage: dict[str, Any] | None = None
    buffer = ""
    status = 200
    try:
        async with client.stream("POST", CHAT_PATH, json=payload) as r:
            status = r.status_code
            if status != 200:
                body = await r.aread()
                yield body
                _record(status, time.perf_counter() - t0, None)
                return
            async for chunk in r.aiter_bytes():
                if first:
                    TTFT.observe(time.perf_counter() - t0)
                    first = False
                yield chunk
                # Parse a decoded copy to pull out the usage block (the final
                # SSE event carries it when stream_options.include_usage is set).
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    found = _usage_from_event(event)
                    if found:
                        usage = found
        elapsed = time.perf_counter() - t0
        _record(status, elapsed, usage)
        _log_langfuse(tracker, payload, None, usage, elapsed)
    finally:
        INFLIGHT.dec()


def _usage_from_event(event: str) -> dict[str, Any] | None:
    for line in event.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
            if data and data != "[DONE]":
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("usage"):
                    return obj["usage"]
    return None
