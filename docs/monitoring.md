# Monitoring

Two layers:

- **Grafana Cloud** for the four operational dimensions (latency, throughput, cost, error rate),
  fed by Prometheus metrics pushed from an OpenTelemetry Collector sidecar.
- **Langfuse** for per-request tracing with token counts and cost.

## Why push, not scrape

The service scales to zero, so a Prometheus server has nothing to scrape when it's idle, and Cloud
Run instances aren't directly addressable anyway. Instead, an **OTel Collector sidecar** lives in
the same instance, scrapes vLLM (`localhost:8000/metrics`) and the gateway
(`localhost:8080/metrics`), and **remote-writes** to Grafana Cloud. The `container-dependencies`
annotation makes the collector start first and stop last, so the app's metrics are scraped right up
to shutdown. The last scrape interval before scale-down can still be lost, which is acceptable for
this workload.

Config: `monitoring/otel-collector.yaml` (mounted into the sidecar from a Secret).

## Dashboard

`monitoring/grafana-dashboard.json`. Import it in Grafana (Dashboards → New → Import → upload), pick
your Prometheus data source when prompted. Panels are grouped by dimension:

| Dimension | Panels (metric) |
|-----------|-----------------|
| Latency | e2e p50/p95/p99 (`vllm:e2e_request_latency_seconds`), TTFT (`vllm:time_to_first_token_seconds`), inter-token (`vllm:time_per_output_token_seconds`) |
| Throughput | tokens/sec (`vllm:generation_tokens_total`), req/sec (`vllm:e2e_request_latency_seconds_count`), running vs waiting (`vllm:num_requests_running` / `:num_requests_waiting`), KV-cache (`vllm:kv_cache_usage_perc`) |
| Cost | $/1M tokens and $/request (derived from a `gpu_cost_per_hour` dashboard variable ÷ throughput), tokens per GPU-hour |
| Error rate | 4xx and 5xx rate (`gateway_requests_total`), request rate by status |

> Note on metric names: this uses the current vLLM **V1** names. The KV-cache gauge is
> `vllm:kv_cache_usage_perc` (the older V0 name `vllm:gpu_cache_usage_perc` was renamed). If a panel
> is empty, curl your vLLM `/metrics` and check the exact name your version emits.

The `gpu_cost_per_hour` dashboard variable (default 1.65) drives the cost panels; set it to your
real instance cost.

## Alerts

`monitoring/alerts.yaml` defines the two required alerts. Either provision the file (after filling
in your Prometheus data source UID) or recreate them in the UI with these expressions:

**High p99 latency** (`for: 5m`):
```promql
histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[5m]))) > 10
```

**KV-cache saturation** (`for: 5m`) — early OOM / preemption signal:
```promql
max(vllm:kv_cache_usage_perc) > 0.9
```

Tune the 10s threshold to your latency SLO.

## Langfuse (tokens + cost per request)

The gateway logs each chat completion to Langfuse with the prompt, the output, token usage, and
latency (best-effort: if Langfuse is down or misconfigured it disables itself and never breaks a
request). Enable it by setting `ENABLE_LANGFUSE=true` and the Langfuse keys.

**Cost attribution for a self-hosted model.** Langfuse computes cost as token counts × a model
price. Qwen has no built-in price, so register one: Langfuse → Settings → Models → New model, with a
`match_pattern` like `(?i)^qwen2\.5-7b-instruct$` and per-token input/output prices derived from
your GPU $/hr ÷ throughput. After that, every traced request shows an attributed cost.

A single Langfuse trace showing one request's tokens + cost satisfies the monitoring proof point
alongside the Grafana view.
