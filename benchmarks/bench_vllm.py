"""Benchmark a running OpenAI-compatible endpoint (vLLM) under fixed concurrency.

Used for both the "vLLM FP16" and "vLLM + AWQ INT4" configs: launch vLLM with
the config you want, then point this client at it. It streams every request so
we can measure time-to-first-token as well as end-to-end latency.

Example (run from the repo root, after starting vLLM on :8000):
    python -m benchmarks.bench_vllm --config vllm-awq --concurrency 16 --num-requests 64

Model-weights memory: read the "Model weights take X.XX GiB" line vLLM prints at
startup and pass it as --weights-gib (vLLM pre-reserves the KV cache, so runtime
nvidia-smi doesn't reflect the weights difference we care about).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx

from benchmarks.common import (
    PROMPTS,
    RequestResult,
    nvidia_smi_used_mib,
    save_summary,
    summarize,
)


async def one_request(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    prompt_tokens = 0
    output_tokens = 0
    content_chunks = 0
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            if r.status_code != 200:
                return RequestResult(ok=False, e2e_s=time.perf_counter() - t0)
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    continue
                obj = json.loads(data)
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    content_chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                    output_tokens = obj["usage"].get("completion_tokens", 0)
    except Exception:
        return RequestResult(ok=False, e2e_s=time.perf_counter() - t0)

    # fall back to chunk count if the server didn't send a usage block
    if output_tokens == 0:
        output_tokens = content_chunks
    return RequestResult(
        ok=True,
        e2e_s=time.perf_counter() - t0,
        ttft_s=ttft,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


async def run_load(
    base_url: str, model: str, num_requests: int, concurrency: int, max_tokens: int, warmup: int
) -> tuple[list[RequestResult], float]:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(300.0, connect=10.0)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:
        # warmup, not measured
        await asyncio.gather(
            *(
                one_request(client, model, PROMPTS[i % len(PROMPTS)], max_tokens)
                for i in range(warmup)
            )
        )

        async def worker(prompt: str) -> RequestResult:
            async with sem:
                return await one_request(client, model, prompt, max_tokens)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - t0

    return list(results), wall


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM endpoint concurrency benchmark")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--config", required=True, help="label for results, e.g. vllm-awq or vllm-fp16")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--cost-per-hour", type=float, default=1.65, help="USD/hr for the instance")
    ap.add_argument(
        "--weights-gib", type=float, default=None, help="model weights GiB from vLLM log"
    )
    args = ap.parse_args()

    results, wall = asyncio.run(
        run_load(
            args.base_url,
            args.model,
            args.num_requests,
            args.concurrency,
            args.max_tokens,
            args.warmup,
        )
    )

    summary = summarize(
        args.config,
        results,
        wall,
        concurrency=args.concurrency,
        cost_per_hour_usd=args.cost_per_hour,
        peak_gpu_mem_gib=args.weights_gib,
    )
    summary.extra["nvidia_smi_used_mib_under_load"] = nvidia_smi_used_mib()
    path = save_summary(summary)
    print(
        f"\n{args.config}: {summary.requests_per_s} req/s, {summary.output_tokens_per_s} tok/s, "
        f"p95 {summary.latency_p95_s}s, ttft-p95 {summary.ttft_p95_s}s, errors {summary.errors}"
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
