# Benchmarks

A reproducible before/after comparison across three configs on the **same NVIDIA L4 (24 GB)**:

| Config | What changes |
|--------|--------------|
| Baseline | HF Transformers `generate`, FP16, one request at a time, no batching |
| vLLM (FP16) | continuous batching + PagedAttention |
| vLLM + AWQ INT4 | 4-bit weight quantization |

Everything uses the same fixed prompt set (`benchmarks/common.py`), excludes warmup, and is run on
the same hardware. The scripts run on a GPU box (the L4), not your laptop.

## Setup on the GPU box

```bash
pip install -r requirements-bench.txt   # torch/vllm already in the vllm image
```

## 1. Baseline (HF Transformers FP16)

Self-contained: loads the model and generates sequentially.

```bash
python -m benchmarks.baseline_hf --num-requests 32 --max-tokens 256
```

Records p50/p95/p99 latency, TTFT (via a streamer), tokens/sec, and **peak GPU memory**
(`torch.cuda.max_memory_allocated`, the real peak for FP16 weights + activations). Writes
`benchmarks/results/baseline-hf-fp16.json`.

## 2. vLLM (FP16) and 3. vLLM + AWQ INT4

These benchmark a running vLLM server under fixed concurrency, so launch vLLM with the config you
want, then point the client at it.

**vLLM FP16** (full-precision weights):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --dtype float16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000
# note the "Model loading took X GiB" line it logs at startup

python -m benchmarks.bench_vllm --config vllm-fp16 --concurrency 16 --num-requests 64 --weights-gib <X>
```

**vLLM + AWQ INT4**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000
# again note "Model loading took X GiB" (expect ~5.3)

python -m benchmarks.bench_vllm --config vllm-awq --concurrency 16 --num-requests 64 --weights-gib <X>
```

The client streams every request, so it measures TTFT as well as end-to-end latency, plus
requests/sec and tokens/sec at the given concurrency. Token counts come from the `usage` block vLLM
returns.

### On the memory number

vLLM pre-reserves the KV cache up front (per `--gpu-memory-utilization`), so runtime `nvidia-smi`
shows a near-constant ~90% and does NOT reflect the weights difference we care about. The honest
comparison is **model-weights memory**, which vLLM logs at startup (`Model loading took X GiB`).
Pass that as `--weights-gib`. Measured here: FP16 ~14.3 GiB, AWQ INT4 ~5.3 GiB. The baseline's
`torch` peak (~14.2 GiB) is the FP16 weights plus a bit of activation.

## Accuracy check (quantization preserves quality)

Run a small GSM8K subset against the FP16 and AWQ endpoints and compare exact-match accuracy:

```bash
python -m benchmarks.accuracy --config vllm-fp16 --num-questions 40   # against the FP16 server
python -m benchmarks.accuracy --config vllm-awq  --num-questions 40   # against the AWQ server
```

In this run AWQ held up fine: 87.5% vs FP16's 80.0% on 40 questions, so no measurable drop (the gap
is within noise at this sample size).

## Aggregate and chart

```bash
python -m benchmarks.plot
```

Reads every JSON in `benchmarks/results/`, writes `summary.md` (the table below) and
`comparison.png` (throughput / latency / cost / memory bars).

## Results

Measured on one `g2-standard-8` (NVIDIA L4, 24 GB) in `asia-southeast1`, same prompt set, warmup
excluded. vLLM configs run at concurrency 16; the baseline is concurrency 1 (HF `generate` doesn't
batch).

| Config | req/s | tok/s | p50 (s) | p95 (s) | p99 (s) | TTFT p95 (s) | Peak mem (GiB) | $/1M tok | GSM8K acc |
|--------|-------|-------|---------|---------|---------|--------------|----------------|----------|-----------|
| baseline-hf-fp16 | 0.082 | 16.5 | 15.436 | 15.468 | 15.474 | 0.072 | 14.21 | 14.2752 | - |
| vllm-fp16 | 1.154 | 233.2 | 15.433 | 15.497 | 15.624 | 0.286 | 14.29 | 1.0125 | 80.0% |
| vllm-awq | 2.735 | 555.6 | 6.241 | 6.554 | 6.751 | 0.278 | 5.29 | 0.425 | 87.5% |

![comparison](img/benchmark.png)

What it shows:

- **vLLM vs the naive baseline: about 14x the throughput** (16.5 -> 233 tok/s) on the same FP16
  weights, purely from continuous batching. Per-request latency is similar, but it serves 16 at once.
- **AWQ on top: 2.4x faster again and about 2.7x less memory** (233 -> 556 tok/s, 14.3 -> 5.3 GiB of
  weights), which also drops p95 latency from 15.5s to 6.6s.
- **End to end: roughly 34x the throughput and 1/33 the cost** of the baseline ($14.28 -> $0.43 per
  1M tokens, at the $0.85/hr this instance runs).
- **Quantizing didn't cost accuracy:** AWQ scored 87.5% on a 40-question GSM8K subset vs FP16's
  80.0% (a 3-question gap, within noise at n=40).

## Method notes (read these honestly)

- Fixed prompt set, identical across configs; warmup requests excluded.
- Cost per 1M tokens is derived: `(instance $/hr ÷ 3600) ÷ tokens/sec × 1e6`. The numbers above use
  `--cost-per-hour 0.85`, about what a `g2-standard-8` (8 vCPU + L4) costs on demand; adjust to your
  real config.
- The baseline is concurrency 1 by design (HF `generate` doesn't batch), so its throughput is what a
  naive deployment gets. vLLM's win comes from continuous batching at concurrency.
- Numbers depend on prompt length, `max_tokens`, and concurrency. Document the exact command next to
  any number you report.
