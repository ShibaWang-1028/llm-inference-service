# Benchmarks

A reproducible before→after comparison across three configs on the **same NVIDIA L4 (24 GB)**:

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
# note the "Model weights take X.XX GiB" line it prints at startup

python -m benchmarks.bench_vllm --config vllm-fp16 --concurrency 16 --num-requests 64 --weights-gib <X>
```

**vLLM + AWQ INT4**:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000
# again note "Model weights take X.XX GiB" (expect ~5.6)

python -m benchmarks.bench_vllm --config vllm-awq --concurrency 16 --num-requests 64 --weights-gib <X>
```

The client streams every request, so it measures TTFT as well as end-to-end latency, plus
requests/sec and tokens/sec at the given concurrency. Token counts come from the `usage` block vLLM
returns.

### On the memory number

vLLM pre-reserves the KV cache up front (per `--gpu-memory-utilization`), so runtime `nvidia-smi`
shows a near-constant ~90% and does NOT reflect the weights difference we care about. The honest
comparison is **model-weights memory**, which vLLM prints at startup (`Model weights take X.XX
GiB`). Pass that as `--weights-gib`. FP16 weights are ~15 GiB; AWQ INT4 ~5.6 GiB. The baseline's
`torch` peak (~16-18 GiB) is the FP16 weights plus activations.

## Accuracy check (quantization preserves quality)

Run a small GSM8K subset against the FP16 and AWQ endpoints and compare exact-match accuracy:

```bash
python -m benchmarks.accuracy --config vllm-fp16 --num-questions 40   # against the FP16 server
python -m benchmarks.accuracy --config vllm-awq  --num-questions 40   # against the AWQ server
```

We expect < 2% absolute drop from quantization.

## Aggregate and chart

```bash
python -m benchmarks.plot
```

Reads every JSON in `benchmarks/results/`, writes `summary.md` (the table below) and
`comparison.png` (throughput / latency / cost / memory bars).

## Results

Paste the contents of `benchmarks/results/summary.md` here after running:

| Config | req/s | tok/s | p50 (s) | p95 (s) | p99 (s) | TTFT p95 (s) | Peak mem (GiB) | $/1M tok | GSM8K acc |
|--------|-------|-------|---------|---------|---------|--------------|----------------|----------|-----------|
| baseline-hf-fp16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| vllm-fp16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| vllm-awq | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Method notes (read these honestly)

- Fixed prompt set, identical across configs; warmup requests excluded.
- Cost per 1M tokens is derived: `(instance $/hr ÷ 3600) ÷ tokens/sec × 1e6`. The default
  `--cost-per-hour 1.65` is an 8 vCPU + 32 GiB + L4 (no zonal redundancy) instance; adjust to your
  real config.
- The baseline is concurrency 1 by design (HF `generate` doesn't batch), so its throughput is what a
  naive deployment gets. vLLM's win comes from continuous batching at concurrency.
- Numbers depend on prompt length, `max_tokens`, and concurrency. Document the exact command next to
  any number you report.
