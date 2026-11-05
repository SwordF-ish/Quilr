"""Task 4 - LLM Gateway: per-tenant token rate limiting + primary/secondary model failover.

    POST /v1/chat/completions   (Authorization: Bearer <tenant api key>)
      1. authenticate tenant (SHA-256 key lookup in SQLite)
      2. estimate tokens, reserve them in the sliding window (or 429)
      3. route: primary (3000 ms deadline, 429 -> fallback) -> secondary
      4. reconcile reservation with provider-reported usage (or refund on failure)

Every error - ours, validation, upstream, or an unexpected crash - is returned in
one envelope and never includes upstream bodies, stack traces, hostnames or
provider names:

    {"error": {"type": "...", "code": "...", "message": "...", "request_id": "req_..."}}
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from rate_limiter import DEFAULT_TOKENS_PER_MINUTE, SQLiteTokenRateLimiter, Tenant, estimate_tokens
from router import AllProvidersFailed, ModelRouter, ProviderConfig, UpstreamClientError

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [llm-router] %(message)s")
log = logging.getLogger("llm-router")

DEFAULT_MAX_TOKENS = 1024


# --- standardized errors --------------------------------------------------------
class GatewayError(Exception):
    def __init__(self, status: int, type_: str, code: str, message: str, headers: dict[str, str] | None = None):
        self.status, self.type, self.code, self.message, self.headers = status, type_, code, message, headers or {}


def error_response(request: Request, status: int, type_: str, code: str, message: str,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:16]}"
    return JSONResponse(
        {"error": {"type": type_, "code": code, "message": message, "request_id": rid}},
        status_code=status,
        headers={"X-Request-ID": rid, **(headers or {})},
    )


# --- request schema --------------------------------------------------------------
class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(pattern=r"^(system|user|assistant|tool|developer)$")
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature, top_p, ... pass through
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)
    stream: bool = False


def create_app(
    db_path: str | None = None,
    primary: ProviderConfig | None = None,
    secondary: ProviderConfig | None = None,
    http_client: httpx.AsyncClient | None = None,
    primary_timeout_s: float = 3.0,
    seed_demo_tenant: bool | None = None,
    limiter: SQLiteTokenRateLimiter | None = None,
) -> FastAPI:
    env = os.environ.get
    limiter = limiter or SQLiteTokenRateLimiter(db_path or env("GATEWAY_DB_PATH", "data/gateway.sqlite3"))
    primary = primary or ProviderConfig("primary", env("PRIMARY_BASE_URL", "http://127.0.0.1:9201/flaky/v1"),
                                        env("PRIMARY_API_KEY", "mock"), env("PRIMARY_MODEL", "primary-large"))
    secondary = secondary or ProviderConfig("secondary", env("SECONDARY_BASE_URL", "http://127.0.0.1:9201/ok/v1"),
                                            env("SECONDARY_API_KEY", "mock"), env("SECONDARY_MODEL", "backup-small"))
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0),
                                              limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
    router = ModelRouter(primary, secondary, client, primary_timeout_s=primary_timeout_s)

    if seed_demo_tenant is None:
        seed_demo_tenant = env("GATEWAY_SEED_DEMO", "1") == "1"
    if seed_demo_tenant and limiter.tenant_count() == 0:
        limiter.upsert_tenant("demo-tenant", env("DEMO_TENANT_KEY", "sk-demo-tenant"), DEFAULT_TOKENS_PER_MINUTE)
        log.info("seeded demo tenant (key: DEMO_TENANT_KEY or 'sk-demo-tenant')")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def janitor():
            while True:
                await asyncio.sleep(30)
                with suppress(Exception):
                    n = await asyncio.to_thread(limiter.purge_expired)
                    if n:
                        log.info("evicted %d expired usage rows", n)

        task = asyncio.create_task(janitor())
        yield
        task.cancel()
        if http_client is None:
            await client.aclose()

    app = FastAPI(title="LLM Gateway - rate limiting & model fallback", lifespan=lifespan)
    app.state.limiter = limiter

    # Outermost safety net: anything unexpected becomes a sanitised 500.
    @app.middleware("http")
    async def request_id_and_crash_guard(request: Request, call_next):
        request.state.request_id = f"req_{uuid.uuid4().hex[:16]}"
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error request_id=%s", request.state.request_id)
            return error_response(request, 500, "internal_error", "internal_error", "An internal error occurred.")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        return error_response(request, exc.status, exc.type, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Field paths only - never echo the submitted values back.
        fields = sorted({".".join(str(p) for p in e["loc"] if p != "body") for e in exc.errors()})
        return error_response(request, 400, "invalid_request_error", "invalid_request",
                              f"Invalid request body. Problem fields: {', '.join(fields) or 'body'}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return error_response(request, exc.status_code, "invalid_request_error", f"http_{exc.status_code}",
                              "Not found." if exc.status_code == 404 else "Request could not be processed.")

    async def authenticate(request: Request) -> Tenant:
        scheme, _, key = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            raise GatewayError(401, "authentication_error", "missing_api_key", "Provide 'Authorization: Bearer <api key>'.")
        tenant = await asyncio.to_thread(limiter.lookup_tenant, key.strip())
        if tenant is None:
            raise GatewayError(401, "authentication_error", "invalid_api_key", "Invalid API key.")
        return tenant

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, req: ChatRequest, tenant: Tenant = Depends(authenticate)):
        # `authenticate` runs as a dependency, i.e. BEFORE body validation:
        # unauthenticated callers get 401, never schema details.
        if req.stream:
            raise GatewayError(400, "invalid_request_error", "stream_not_supported",
                               "This endpoint does not support stream=true.")

        body = req.model_dump(exclude_none=True)
        body["max_tokens"] = req.max_tokens or DEFAULT_MAX_TOKENS
        est = estimate_tokens(body["messages"], body["max_tokens"])

        decision = await asyncio.to_thread(limiter.try_acquire, tenant.tenant_id, est, tenant.tokens_per_minute)
        rl_headers = {"X-RateLimit-Limit-Tokens": str(decision.limit),
                      "X-RateLimit-Remaining-Tokens": str(decision.remaining)}
        if not decision.allowed:
            if decision.retry_after_s is None:
                raise GatewayError(413, "invalid_request_error", "request_too_large",
                                   f"Request needs ~{est} tokens, above the per-minute limit of {decision.limit}.",
                                   rl_headers)
            raise GatewayError(429, "rate_limit_error", "tenant_rate_limited",
                               "Token rate limit exceeded for this API key. Retry later.",
                               {**rl_headers, "Retry-After": str(max(1, math.ceil(decision.retry_after_s)))})

        settled = False
        try:
            result = await router.complete(body)
            usage = result.body.get("usage") or {}
            actual = usage.get("total_tokens")
            if isinstance(actual, int) and actual >= 0:
                await asyncio.to_thread(limiter.reconcile, decision.reservation_id, actual)
            settled = True
        except AllProvidersFailed as exc:
            log.error("all providers failed tenant=%s request_id=%s attempts=%s",
                      tenant.tenant_id, request.state.request_id, exc.attempts)
            raise GatewayError(503, "upstream_error", "all_models_unavailable",
                               "The model service is temporarily unavailable. Please retry.",
                               {"Retry-After": "5"}) from None
        except UpstreamClientError as exc:
            log.warning("upstream rejected request status=%s request_id=%s", exc.status, request.state.request_id)
            raise GatewayError(400, "invalid_request_error", "upstream_rejected_request",
                               "The model rejected this request.") from None
        finally:
            if not settled:  # failure or client disconnect (CancelledError): refund the reservation
                with suppress(Exception):
                    await asyncio.shield(asyncio.to_thread(limiter.release, decision.reservation_id))

        out = result.body
        headers = {**rl_headers, "X-Gateway-Fallback": "true" if result.fallback_used else "false"}
        return JSONResponse(out, headers=headers)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9200")))
