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

    # Match the demo UI: warm off-white, ink text, terracotta accent. Bars go
    # gray -> light terracotta -> accent, so "more optimized" reads as "more accent".
    bg, ink, line = "#fbfaf7", "#191917", "#e3ded4"
    color = {"baseline-hf-fp16": "#ccc7ba", "vllm-fp16": "#d98a5f", "vllm-awq": "#b15a39"}
    nice = {
        "baseline-hf-fp16": "Baseline\nHF FP16",
        "vllm-fp16": "vLLM\nFP16",
        "vllm-awq": "vLLM\n+ AWQ",
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "text.color": ink,
            "xtick.color": ink,
        }
    )

    labels = [nice.get(s["config"], s["config"]) for s in summaries]
    colors = [color.get(s["config"], "#999999") for s in summaries]
    panels = [
        ("Throughput, tokens/sec (higher is better)", "output_tokens_per_s", lambda v: f"{v:.0f}"),
        ("p95 latency, seconds (lower is better)", "latency_p95_s", lambda v: f"{v:.1f}s"),
        (
            "Cost, USD / 1M tokens (lower is better)",
            "cost_per_1m_tokens_usd",
            lambda v: f"${v:.2f}",
        ),
        ("Model weights, GiB (lower is better)", "peak_gpu_mem_gib", lambda v: f"{v:.1f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, (title, key, fmt) in zip(axes.flat, panels, strict=False):
        values = [(s.get(key) or 0) for s in summaries]
        bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
        ax.set_title(title, fontsize=11.5, loc="left", pad=12, color=ink)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(line)
        ax.tick_params(length=0, labelsize=11)
        ax.set_yticks([])
        ax.margins(y=0.22)
        for b, v in zip(bars, values, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                fmt(v),
                ha="center",
                va="bottom",
                fontsize=12,
                color=ink,
            )

    fig.suptitle(
        "Qwen2.5-7B on one NVIDIA L4: baseline vs vLLM vs vLLM + AWQ",
        fontsize=15,
        y=0.99,
        color=ink,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.6, w_pad=3.0)
    out = RESULTS_DIR / "comparison.png"
    fig.savefig(out, dpi=150, facecolor=bg)
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
