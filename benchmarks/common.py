"""Shared helpers for the benchmark scripts.

Kept dependency-light: percentiles are pure Python, charts import matplotlib
lazily. The numbers here feed docs/benchmarks.md.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# A fixed, varied prompt set so every config is measured on the same inputs.
PROMPTS: list[str] = [
    "Explain what a transformer model is in two sentences.",
    "Write a haiku about distributed systems.",
    "What is the difference between latency and throughput?",
    "Give me three tips for writing clean Python.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Translate 'good morning, how are you?' into French and Japanese.",
    "What are the tradeoffs of microservices vs a monolith?",
    "Write a SQL query to find the second highest salary in a table.",
    "Explain quantization of neural networks to a beginner.",
    "List five common HTTP status codes and what they mean.",
    "What is PagedAttention and why does it help LLM serving?",
    "Describe the CAP theorem briefly.",
    "Write a short cover-letter opening for a software job.",
    "How does a hash map work under the hood?",
    "Give a recipe for a simple tomato pasta.",
    "What is the purpose of a load balancer?",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


@dataclass
class RequestResult:
    ok: bool
    e2e_s: float = 0.0
    ttft_s: float | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Summary:
    config: str
    requests: int = 0
    errors: int = 0
    concurrency: int = 1
    wall_s: float = 0.0
    requests_per_s: float = 0.0
    output_tokens_per_s: float = 0.0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    latency_p99_s: float = 0.0
    ttft_p50_s: float | None = None
    ttft_p95_s: float | None = None
    peak_gpu_mem_gib: float | None = None
    cost_per_1m_tokens_usd: float | None = None
    extra: dict = field(default_factory=dict)


def summarize(
    config: str,
    results: list[RequestResult],
    wall_s: float,
    concurrency: int,
    cost_per_hour_usd: float,
    peak_gpu_mem_gib: float | None = None,
) -> Summary:
    ok = [r for r in results if r.ok]
    e2e = [r.e2e_s for r in ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    out_tokens = sum(r.output_tokens for r in ok)

    tps = out_tokens / wall_s if wall_s > 0 else 0.0
    cost = cost_per_million_tokens(tps, cost_per_hour_usd) if tps > 0 else None

    return Summary(
        config=config,
        requests=len(results),
        errors=len(results) - len(ok),
        concurrency=concurrency,
        wall_s=round(wall_s, 3),
        requests_per_s=round(len(ok) / wall_s, 3) if wall_s > 0 else 0.0,
        output_tokens_per_s=round(tps, 1),
        latency_p50_s=round(percentile(e2e, 50), 3),
        latency_p95_s=round(percentile(e2e, 95), 3),
        latency_p99_s=round(percentile(e2e, 99), 3),
        ttft_p50_s=round(percentile(ttft, 50), 3) if ttft else None,
        ttft_p95_s=round(percentile(ttft, 95), 3) if ttft else None,
        peak_gpu_mem_gib=peak_gpu_mem_gib,
        cost_per_1m_tokens_usd=round(cost, 4) if cost is not None else None,
    )


def cost_per_million_tokens(tokens_per_s: float, cost_per_hour_usd: float) -> float:
    # $/1M tokens = ($/hour) / (tokens/hour) * 1e6
    tokens_per_hour = tokens_per_s * 3600.0
    return cost_per_hour_usd * 1_000_000.0 / tokens_per_hour


def nvidia_smi_used_mib(gpu_index: int = 0) -> float | None:
    """Best-effort: total GPU memory in use right now, via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            text=True,
            timeout=10,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def save_summary(summary: Summary) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{summary.config}.json"
    path.write_text(json.dumps(summary.__dict__, indent=2))
    return path


def load_summaries() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out
