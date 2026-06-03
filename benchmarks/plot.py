"""Aggregate the result JSONs in results/ into a comparison chart + markdown table.

python -m benchmarks.plot
"""

from __future__ import annotations

import json

from benchmarks.common import RESULTS_DIR, load_summaries

# preferred left-to-right order in tables/charts
ORDER = ["baseline-hf-fp16", "vllm-fp16", "vllm-awq"]


def _order_key(config: str) -> int:
    return ORDER.index(config) if config in ORDER else len(ORDER)


def load_accuracy() -> dict[str, float]:
    out: dict[str, float] = {}
    for p in RESULTS_DIR.glob("accuracy-*.json"):
        d = json.loads(p.read_text())
        out[d["config"]] = d["accuracy"]
    return out


def make_chart(summaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s["config"] for s in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("Output tokens/sec (higher is better)", "output_tokens_per_s"),
        ("p95 latency, seconds (lower is better)", "latency_p95_s"),
        ("Cost per 1M tokens, USD (lower is better)", "cost_per_1m_tokens_usd"),
        ("Model weights / peak GPU memory, GiB", "peak_gpu_mem_gib"),
    ]
    for ax, (title, key) in zip(axes.flat, panels, strict=False):
        values = [(s.get(key) or 0) for s in summaries]
        ax.bar(labels, values, color=["#9aa4b2", "#6f8fd6", "#4f8cff"][: len(labels)])
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelrotation=15)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Qwen2.5-7B inference: baseline vs vLLM vs vLLM+AWQ", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "comparison.png"
    fig.savefig(out, dpi=130)
    print(f"chart -> {out}")


def make_table(summaries: list[dict], accuracy: dict[str, float]) -> None:
    head = (
        "| Config | req/s | tok/s | p50 (s) | p95 (s) | p99 (s) | TTFT p95 (s) | "
        "Peak mem (GiB) | $/1M tok | GSM8K acc |"
    )
    sep = "|" + "---|" * 10
    rows = [head, sep]
    for s in summaries:
        acc = accuracy.get(s["config"])
        rows.append(
            f"| {s['config']} | {s['requests_per_s']} | {s['output_tokens_per_s']} | "
            f"{s['latency_p50_s']} | {s['latency_p95_s']} | {s['latency_p99_s']} | "
            f"{s.get('ttft_p95_s')} | {s.get('peak_gpu_mem_gib')} | "
            f"{s.get('cost_per_1m_tokens_usd')} | {f'{acc:.1%}' if acc is not None else '-'} |"
        )
    table = "\n".join(rows) + "\n"
    out = RESULTS_DIR / "summary.md"
    out.write_text(table)
    print(f"table -> {out}\n")
    print(table)


def main() -> None:
    summaries = [s for s in load_summaries() if "output_tokens_per_s" in s]
    summaries.sort(key=lambda s: _order_key(s["config"]))
    if not summaries:
        print("No benchmark results found in results/. Run the benchmarks first.")
        return
    accuracy = load_accuracy()
    make_table(summaries, accuracy)
    try:
        make_chart(summaries)
    except Exception as e:  # matplotlib optional / headless quirks
        print(f"(skipped chart: {e})")


if __name__ == "__main__":
    main()
