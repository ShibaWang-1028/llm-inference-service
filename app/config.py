"""App configuration, loaded from environment variables / a local .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        # we use model_id / model_name as field names, so turn off pydantic's
        # protection of the "model_" attribute namespace
        protected_namespaces=(),
    )

    # Upstream vLLM (OpenAI-compatible) server. In production this is the local
    # vLLM process; in dev it's the fake upstream in tools/fake_vllm.py.
    upstream_base_url: str = "http://127.0.0.1:8000"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    model_name: str = "Qwen2.5-7B-Instruct"

    # Auth + rate limiting
    api_keys: str = ""  # comma-separated list of allowed keys; empty = auth off
    # A public demo key the UI uses automatically so visitors don't type anything.
    # It still counts as a valid key (auth stays real), but it's rate-limited per
    # client IP instead of per key. Leave empty to not ship a public demo key.
    demo_api_key: str = ""
    rate_limit: str = "60/minute"

    # Timeouts / server
    request_timeout_s: float = 120.0
    port: int = 8080
    log_level: str = "info"

    # Langfuse (per-request token + cost tracking)
    enable_langfuse: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # OpenTelemetry traces (-> Grafana Cloud Tempo)
    enable_otel: bool = False
    otel_service_name: str = "llm-inference-gateway"
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""

    @property
    def allowed_keys(self) -> set[str]:
        keys = {k.strip() for k in self.api_keys.split(",") if k.strip()}
        if self.demo_api_key.strip():
            keys.add(self.demo_api_key.strip())
        return keys

    @property
    def auth_enabled(self) -> bool:
        return len(self.allowed_keys) > 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
