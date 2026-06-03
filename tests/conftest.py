"""Test fixtures.

Each test gets a gateway app whose upstream client is wired, in-process, to the
fake vLLM server via httpx's ASGI transport. No network, no GPU.
"""

from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest_asyncio

from app.config import Settings
from app.main import create_app
from app.telemetry import LangfuseTracker
from tools.fake_vllm import app as fake_app

ClientFactory = Callable[..., Awaitable[httpx.AsyncClient]]


@pytest_asyncio.fixture
async def client_factory() -> AsyncIterator[ClientFactory]:
    created: list[tuple[httpx.AsyncClient, object]] = []

    async def _factory(**overrides: object) -> httpx.AsyncClient:
        params: dict[str, object] = {
            "api_keys": "testkey",
            "rate_limit": "1000/minute",
            "enable_langfuse": False,
            "enable_otel": False,
        }
        params.update(overrides)
        # _env_file=None ignores any local .env so tests are deterministic
        settings = Settings(_env_file=None, **params)
        app = create_app(settings)
        # ASGI transport doesn't run lifespan, so wire state by hand.
        app.state.settings = settings
        app.state.upstream = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream"
        )
        app.state.tracker = LangfuseTracker(settings)
        gw = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")
        created.append((gw, app))
        return gw

    yield _factory

    for gw, app in created:
        await gw.aclose()
        await app.state.upstream.aclose()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def client(client_factory: ClientFactory) -> httpx.AsyncClient:
    return await client_factory()


AUTH = {"Authorization": "Bearer testkey"}
