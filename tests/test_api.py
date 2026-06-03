"""Gateway API tests, run against the fake vLLM upstream."""

import httpx

from tests.conftest import AUTH

CHAT = "/v1/chat/completions"
PROMPT = {"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hello there"}]}


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
