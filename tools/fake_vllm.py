"""A fake vLLM OpenAI server for local development and tests.

It speaks just enough of vLLM's OpenAI-compatible API for the gateway, the demo
UI, and the test suite to run with no GPU and no real model:

  POST /v1/chat/completions   streaming (SSE) and non-streaming
  GET  /health                readiness (always ready here)
  GET  /v1/models             single-model list
  GET  /metrics               a handful of vLLM-style Prometheus metrics

Run it locally:  uvicorn tools.fake_vllm:app --port 8000
Slow the stream down for a nicer demo:  FAKE_DELAY=0.03 uvicorn tools.fake_vllm:app --port 8000
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

app = FastAPI(title="Fake vLLM upstream")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen2.5-7B-Instruct")
STREAM_DELAY = float(os.environ.get("FAKE_DELAY", "0.0"))

# Rough counters so /metrics shows movement as you exercise the gateway.
STATS = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _estimate_tokens(text: str) -> int:
    # Good enough for a fake: ~1 token per whitespace-delimited word, min 1.
    return max(1, len(text.split()))


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal-style content parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts)
    return ""


def _build_reply(messages: list[dict]) -> str:
    last = _last_user_message(messages).strip() or "your question"
    # markdown so the UI's rendering (lists, bold, inline code) is exercised
    return (
        f"Here is a short take on **{last[:80]}**.\n\n"
        "A few points:\n\n"
        "- vLLM serves with continuous batching and PagedAttention\n"
        "- AWQ stores the weights in 4-bit, which cuts memory a lot\n"
        "- The API is OpenAI-compatible, so `client.chat.completions.create` just works\n\n"
        "That is the gist. (This reply comes from the local fake upstream.)"
    )


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "fake"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", MODEL_NAME)
    stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    prompt_text = " ".join(str(m.get("content", "")) for m in messages)
    prompt_tokens = _estimate_tokens(prompt_text)
    reply = _build_reply(messages)
    completion_tokens = _estimate_tokens(reply)

    STATS["requests"] += 1
    STATS["prompt_tokens"] += prompt_tokens
    STATS["completion_tokens"] += completion_tokens

    created = int(time.time())
    cmpl_id = f"chatcmpl-fake-{STATS['requests']}"

    if not stream:
        return JSONResponse(
            {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    async def event_stream():
        def sse(obj: dict) -> bytes:
            return f"data: {json.dumps(obj)}\n\n".encode()

        base = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }

        # role delta first
        yield sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        for word in reply.split(" "):
            if STREAM_DELAY:
                await asyncio.sleep(STREAM_DELAY)
            yield sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                    ],
                }
            )

        # final choice with finish_reason
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})

        # usage chunk (only when asked), matching vLLM/OpenAI behaviour
        if include_usage:
            yield sse(
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )

        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """A small subset of vLLM-style metrics, with the current V1 names."""
    r = STATS["requests"]
    label = f'{{model_name="{MODEL_NAME}"}}'
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{label} 0",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{label} 0",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{label} 0.12",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total{label} {STATS['prompt_tokens']}",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{label} {STATS['completion_tokens']}",
        "# TYPE vllm:e2e_request_latency_seconds histogram",
        f'vllm:e2e_request_latency_seconds_bucket{{model_name="{MODEL_NAME}",le="0.5"}} {r}',
        f'vllm:e2e_request_latency_seconds_bucket{{model_name="{MODEL_NAME}",le="1.0"}} {r}',
        f'vllm:e2e_request_latency_seconds_bucket{{model_name="{MODEL_NAME}",le="+Inf"}} {r}',
        f"vllm:e2e_request_latency_seconds_sum{label} {round(r * 0.3, 3)}",
        f"vllm:e2e_request_latency_seconds_count{label} {r}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")
