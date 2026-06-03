"""Baseline: Hugging Face Transformers FP16, plain model.generate(), no batching.

This is deliberately the un-optimized path: one request at a time, eager
generation, full FP16 weights. It's the "before" in the before/after story.

Run on the L4 box (needs a GPU), from the repo root:
    python -m benchmarks.baseline_hf --num-requests 32 --max-tokens 256
"""

from __future__ import annotations

import argparse
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from benchmarks.common import PROMPTS, RequestResult, save_summary, summarize

CONFIG = "baseline-hf-fp16"


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    # Transformers v5 renamed torch_dtype -> dtype; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="cuda"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda"
        )
    model.eval()
    return tok, model


def run_one(model, tok, prompt: str, max_tokens: int) -> RequestResult:
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    prompt_tokens = int(inputs.shape[-1])

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    ttft = None
    pieces: list[str] = []
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t0
        pieces.append(chunk)
    thread.join()

    e2e = time.perf_counter() - t0
    output_tokens = len(tok("".join(pieces), add_special_tokens=False).input_ids)
    return RequestResult(
        ok=True, e2e_s=e2e, ttft_s=ttft, prompt_tokens=prompt_tokens, output_tokens=output_tokens
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="HF Transformers FP16 baseline benchmark")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2, help="warmup requests excluded from results")
    ap.add_argument("--cost-per-hour", type=float, default=1.65, help="USD/hr for the instance")
    args = ap.parse_args()

    tok, model = load_model(args.model)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests + args.warmup)]

    # warmup (compiles kernels, fills caches), not measured
    for p in prompts[: args.warmup]:
        run_one(model, tok, p, args.max_tokens)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    results = [run_one(model, tok, p, args.max_tokens) for p in prompts[args.warmup :]]
    wall = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)

    summary = summarize(
        CONFIG,
        results,
        wall,
        concurrency=1,  # no batching in the baseline
        cost_per_hour_usd=args.cost_per_hour,
        peak_gpu_mem_gib=round(peak_gib, 2),
    )
    path = save_summary(summary)
    print(
        f"\n{CONFIG}: {summary.output_tokens_per_s} tok/s, "
        f"p95 {summary.latency_p95_s}s, peak {summary.peak_gpu_mem_gib} GiB"
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
