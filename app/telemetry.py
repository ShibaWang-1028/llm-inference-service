"""Observability: Prometheus metrics, OpenTelemetry traces, Langfuse tracking.

vLLM exposes its own rich engine metrics (latency histograms, TTFT, KV-cache
usage, queue depth) on its own /metrics. The metrics here are *gateway-level*:
request counts by status (for the error-rate panels), gateway latency, TTFT as
seen by the gateway, and token counters. The sidecar collector scrapes both.
"""

import logging
from typing import TYPE_CHECKING, Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from app.config import Settings

logger = logging.getLogger("gateway.telemetry")

# ---- Gateway-level Prometheus metrics ----
REQUESTS = Counter(
    "gateway_requests_total",
    "Total HTTP requests handled by the gateway",
    ["route", "method", "status"],
)
LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end gateway request latency",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)
TTFT = Histogram(
    "gateway_time_to_first_token_seconds",
    "Time to first streamed chunk for streaming requests",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TOKENS = Counter(
    "gateway_tokens_total",
    "Tokens processed, by type",
    ["type"],  # prompt | completion
)
INFLIGHT = Gauge(
    "gateway_inflight_requests",
    "Requests currently being proxied to the upstream",
)


def metrics_response() -> tuple[bytes, str]:
    """Render the Prometheus exposition for the /metrics route."""
    return generate_latest(), CONTENT_TYPE_LATEST


def setup_otel(app: "FastAPI", settings: "Settings") -> None:
    """Wire up OpenTelemetry tracing if enabled. Best-effort: never fatal."""
    if not settings.enable_otel or not settings.otel_exporter_otlp_endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = endpoint + "/v1/traces"

        # Headers come in as "k=v,k2=v2". Grafana Cloud docs tell you to write
        # the auth header with %20 instead of a space (the env-var path URL-
        # decodes it); since we pass it explicitly here, decode it ourselves.
        headers: dict[str, str] = {}
        for pair in settings.otel_exporter_otlp_headers.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                headers[k.strip()] = v.strip().replace("%20", " ")

        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.otel_service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers or None))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        app.state.tracer_provider = provider
        logger.info("OpenTelemetry tracing enabled -> %s", endpoint)
    except Exception as e:  # pragma: no cover - depends on optional extras
        logger.warning("Failed to set up OpenTelemetry, continuing without it: %s", e)


class LangfuseTracker:
    """Best-effort per-request token/cost logging to Langfuse.

    Disabled unless ENABLE_LANGFUSE=true and keys are set. Any failure disables
    it rather than breaking a request. The model name we send must match a model
    definition registered in Langfuse for cost to be attributed (see docs).
    """

    def __init__(self, settings: "Settings") -> None:
        self.enabled = bool(
            settings.enable_langfuse
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )
        self._client: Any = None
        if not self.enabled:
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracking enabled")
        except Exception as e:
            logger.warning("Failed to init Langfuse, disabling: %s", e)
            self.enabled = False

    def log_chat(
        self,
        *,
        model: str | None,
        messages: Any,
        output: str | None,
        usage: dict[str, Any] | None,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._client is None:
            return
        usage = usage or {}
        try:
            # Langfuse Python SDK v4: a generation observation, finalized with end().
            gen = self._client.start_observation(
                name="chat.completions",
                as_type="generation",
                model=model,
                input=messages,
                output=output,
                usage_details={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                metadata={**(metadata or {}), "latency_ms": latency_ms},
            )
            gen.end()
        except Exception as e:
            logger.warning("Langfuse log failed, disabling: %s", e)
            self.enabled = False

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
