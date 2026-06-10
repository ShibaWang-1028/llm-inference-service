"""FastAPI gateway: an OpenAI-compatible API in front of a local vLLM server.

Responsibilities that live here (and not in vLLM): API-key auth, per-key rate
limiting, gateway metrics, request validation, the demo UI, and health probes.
The actual generation is proxied to the upstream vLLM server (see inference.py).
"""

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import inference
from app.config import Settings, get_settings
from app.schemas import ChatCompletionRequest, HealthResponse
from app.telemetry import LangfuseTracker, metrics_response, setup_otel

logger = logging.getLogger("gateway")

UI_FILE = Path(__file__).resolve().parent.parent / "ui" / "index.html"


async def _warm_upstream(app: FastAPI) -> None:
    """Wait for vLLM to load the model, then run one small completion before
    reporting ready. The first inference JIT-compiles a few Triton kernels, so
    without this the first visitor eats that latency spike; with it, "ready"
    means the next reply is fast, not just that the weights are loaded. The
    phase feeds /health/ready so the UI's wake-up screen can show real stages.
    """
    client: httpx.AsyncClient = app.state.upstream
    settings: Settings = app.state.settings
    t0: float = app.state.started_at
    while not await inference.is_upstream_ready(client):
        await asyncio.sleep(2.0)
    loaded_s = time.monotonic() - t0
    app.state.model_phase = "warming"
    try:
        await client.post(
            inference.CHAT_PATH,
            json={
                "model": settings.model_name,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
            },
        )
    except httpx.HTTPError as exc:  # never block readiness on the warmup
        logger.warning("Warmup completion failed: %s", exc)
    app.state.model_phase = "ready"
    app.state.model_ready = True
    logger.info(
        "Cold start: model loaded %.1fs, warmed %.1fs after gateway start",
        loaded_s,
        time.monotonic() - t0,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    def rate_limit_key(request: Request) -> str:
        """Per API key, except the public demo key is limited per client IP so
        one shared demo key doesn't put every visitor in the same bucket."""
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token and token != settings.demo_api_key:
            return token
        return get_remote_address(request)

    limiter = Limiter(key_func=rate_limit_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.upstream = inference.build_upstream_client(
            settings.upstream_base_url, settings.request_timeout_s
        )
        app.state.tracker = LangfuseTracker(settings)
        app.state.started_at = time.monotonic()
        app.state.model_phase = "loading"
        app.state.model_ready = False
        app.state.warm_task = asyncio.create_task(_warm_upstream(app))
        if not settings.auth_enabled:
            logger.warning(
                "API-key auth is DISABLED (no API_KEYS set). OK for local dev, not for prod."
            )
        yield
        # Flush observability before the instance goes away (matters on scale-to-zero).
        app.state.warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.warm_task
        await app.state.upstream.aclose()
        app.state.tracker.flush()
        provider = getattr(app.state, "tracer_provider", None)
        if provider is not None:
            provider.shutdown()

    app = FastAPI(title="LLM Inference Gateway", version="0.1.0", lifespan=lifespan)
    # Public demo API: allow browser calls from any origin (e.g. the UI hosted on
    # Netlify). Auth is a bearer token, not cookies, so credentials stay off.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.limiter = limiter
    # slowapi's handler is typed for its own exception; mypy can't line it up
    # with Starlette's broader signature, but it works fine at runtime.
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    setup_otel(app, settings)

    async def model_ready(request: Request) -> bool:
        """Ready = vLLM answers /health AND the one-off warmup completion ran.
        Unit tests wire app.state by hand (no lifespan), so when the warmup
        machinery is absent fall back to the plain upstream liveness check."""
        state = request.app.state
        live = await inference.is_upstream_ready(state.upstream)
        warmed = getattr(state, "model_ready", None)
        return live if warmed is None else (warmed and live)

    def startup_status(request: Request) -> dict[str, Any]:
        """Phase + uptime for the UI's wake-up screen; empty without lifespan."""
        state = request.app.state
        status: dict[str, Any] = {}
        phase = getattr(state, "model_phase", None)
        if phase is not None:
            status["phase"] = phase
        started = getattr(state, "started_at", None)
        if started is not None:
            status["uptime_s"] = round(time.monotonic() - started, 1)
        return status

    async def require_api_key(request: Request) -> None:
        s: Settings = request.app.state.settings
        if not s.auth_enabled:
            return
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token not in s.allowed_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ---- OpenAI-compatible API ----

    @app.post("/v1/chat/completions")
    @limiter.limit(settings.rate_limit)
    async def chat_completions(request: Request, _: None = Depends(require_api_key)):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": {"message": "invalid JSON body"}}
            )
        try:
            req = ChatCompletionRequest.model_validate(body)
        except ValidationError as e:
            # drop url/context so the error list is JSON-serializable
            details = e.errors(include_url=False, include_context=False)
            return JSONResponse(
                status_code=422,
                content={"error": {"message": "invalid request", "details": details}},
            )

        payload = req.model_dump(exclude_unset=True)
        client = request.app.state.upstream
        tracker = request.app.state.tracker

        # On a cold start the gateway is up before vLLM has loaded the model.
        # Return a clean 503 (the UI shows it as "waking up") instead of erroring.
        if not await model_ready(request):
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "The model is starting up (cold start). Retry in a few seconds.",
                        "type": "model_loading",
                        **startup_status(request),
                    }
                },
            )

        if payload.get("stream"):
            # Ask the upstream to include usage in the final SSE chunk so we can
            # record tokens/cost without breaking the passthrough.
            opts = payload.get("stream_options")
            if not isinstance(opts, dict):
                opts = {}
            opts.setdefault("include_usage", True)
            payload["stream_options"] = opts
            return StreamingResponse(
                inference.stream_chat(client, payload, tracker),
                media_type="text/event-stream",
            )

        status, data = await inference.complete_chat(client, payload, tracker)
        return JSONResponse(status_code=status, content=data)

    @app.get("/v1/models")
    async def list_models(request: Request, _: None = Depends(require_api_key)):
        client = request.app.state.upstream
        try:
            r = await client.get("/v1/models")
            return JSONResponse(status_code=r.status_code, content=r.json())
        except httpx.HTTPError:
            s: Settings = request.app.state.settings
            return {
                "object": "list",
                "data": [{"id": s.model_name, "object": "model", "owned_by": "local"}],
            }

    # ---- Health probes ----

    @app.get("/health/live", response_model=HealthResponse)
    async def health_live() -> HealthResponse:
        # The gateway process is up. Says nothing about the model.
        return HealthResponse(status="alive")

    @app.get("/health/ready")
    async def health_ready(request: Request):
        ready = await model_ready(request)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready", **startup_status(request)},
        )

    # ---- Demo UI config ----

    @app.get("/config")
    async def ui_config(request: Request):
        # Lets the demo page auto-use a public key, so visitors type nothing.
        s: Settings = request.app.state.settings
        return {
            "model": s.model_name,
            "demo_key": s.demo_api_key,
            "auth_required": s.auth_enabled,
        }

    # ---- Metrics + demo UI ----

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        if UI_FILE.exists():
            return FileResponse(UI_FILE)
        return PlainTextResponse("Gateway is running. See /docs for the API.")

    return app


# Module-level app for `uvicorn app.main:app`
app = create_app()
