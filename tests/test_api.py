"""Gateway API tests, run against the fake vLLM upstream."""

import time

import httpx

from app.config import Settings
from app.main import _warm_upstream, create_app
from app.telemetry import LangfuseTracker
from tests.conftest import AUTH
from tools.fake_vllm import app as fake_app

CHAT = "/v1/chat/completions"
PROMPT = {"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hello there"}]}


def _app_with_upstream(transport: httpx.AsyncBaseTransport):
    """An app wired like production's lifespan (warmup state included), with
    the upstream behind the given transport."""
    settings = Settings(_env_file=None, api_keys="testkey")
    app = create_app(settings)
    app.state.settings = settings
    app.state.upstream = httpx.AsyncClient(transport=transport, base_url="http://upstream")
    app.state.tracker = LangfuseTracker(settings)
    app.state.started_at = time.monotonic()
    app.state.model_phase = "loading"
    app.state.model_ready = False
    gw = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")
    return app, gw


async def test_health_live(client: httpx.AsyncClient) -> None:
    r = await client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


async def test_health_ready(client: httpx.AsyncClient) -> None:
    r = await client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


async def test_chat_requires_auth(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=PROMPT)
    assert r.status_code == 401


async def test_chat_rejects_bad_key(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=PROMPT, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


async def test_chat_completion_nonstream(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json=PROMPT, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_chat_completion_stream(client: httpx.AsyncClient) -> None:
    payload = {**PROMPT, "stream": True}
    r = await client.post(CHAT, json=payload, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "data:" in text
    assert "[DONE]" in text
    # the gateway injects include_usage, so the stream must carry a usage block
    assert '"usage"' in text


async def test_invalid_request_returns_422(client: httpx.AsyncClient) -> None:
    r = await client.post(CHAT, json={"model": "x", "messages": []}, headers=AUTH)
    assert r.status_code == 422


async def test_models_list(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "Qwen2.5-7B-Instruct" in ids


async def test_metrics_endpoint(client: httpx.AsyncClient) -> None:
    # drive one request so a counter is non-zero
    await client.post(CHAT, json=PROMPT, headers=AUTH)
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "gateway_requests_total" in r.text


async def test_config_endpoint(client: httpx.AsyncClient) -> None:
    r = await client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "Qwen2.5-7B-Instruct"
    assert body["auth_required"] is True


async def test_demo_key_works_and_is_exposed(client_factory) -> None:
    # public demo: no private keys, just a demo key the UI auto-uses
    c = await client_factory(api_keys="", demo_api_key="try-the-demo")
    cfg = (await c.get("/config")).json()
    assert cfg["demo_key"] == "try-the-demo"
    assert cfg["auth_required"] is True  # the demo key still enables auth

    ok = await c.post(CHAT, json=PROMPT, headers={"Authorization": "Bearer try-the-demo"})
    assert ok.status_code == 200
    nope = await c.post(CHAT, json=PROMPT)  # no key at all
    assert nope.status_code == 401


async def test_auth_disabled_when_no_keys(client_factory) -> None:
    c = await client_factory(api_keys="")
    r = await c.post(CHAT, json=PROMPT)  # no auth header
    assert r.status_code == 200


async def test_rate_limit(client_factory) -> None:
    c = await client_factory(rate_limit="2/minute")
    codes = [(await c.post(CHAT, json=PROMPT, headers=AUTH)).status_code for _ in range(3)]
    assert codes[0] == 200
    assert codes[-1] == 429


async def test_not_ready_when_upstream_down() -> None:
    """vLLM unreachable (cold start): /health/ready is 503 with phase+uptime,
    and chat returns the clean model_loading error instead of blowing up."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app, gw = _app_with_upstream(httpx.MockTransport(refuse))
    try:
        r = await gw.get("/health/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "not_ready"
        assert body["phase"] == "loading"
        assert body["uptime_s"] >= 0

        chat = await gw.post(CHAT, json=PROMPT, headers=AUTH)
        assert chat.status_code == 503
        assert chat.json()["error"]["type"] == "model_loading"
    finally:
        await gw.aclose()
        await app.state.upstream.aclose()


async def test_ready_only_after_warmup() -> None:
    """With vLLM live but the warmup not yet run, readiness stays 503;
    _warm_upstream runs the warmup completion and flips it to ready."""
    app, gw = _app_with_upstream(httpx.ASGITransport(app=fake_app))
    try:
        warming = await gw.get("/health/ready")
        assert warming.status_code == 503
        assert warming.json()["status"] == "not_ready"

        await _warm_upstream(app)
        assert app.state.model_ready is True
        assert app.state.model_phase == "ready"

        ready = await gw.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["phase"] == "ready"
    finally:
        await gw.aclose()
        await app.state.upstream.aclose()
