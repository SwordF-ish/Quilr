"""Task 2 - MCP Security Gateway (reverse proxy with tool-level authorization).

    AI agent ──HTTP JSON-RPC──▶  [ gateway ]  ──HTTP JSON-RPC──▶  downstream MCP server
                 Bearer <jwt>      │  1. authenticate (JWT -> role)
                                   │  2. parse JSON-RPC envelope
                                   │  3. tools/list  -> forward transparently
                                   │     tools/call  -> admin_* requires role=admin
                                   │                    else -32001, downstream never called
                                   │  4. stream downstream response back

Security decisions
------------------
* Tokens are HS256 JWTs; the role is read from the ``role`` claim *after*
  signature + expiry verification. Unknown roles are rejected.
* The client's bearer token is NOT forwarded downstream (no token passthrough,
  per MCP security best practices). Identity is passed as ``X-Gateway-*`` headers.
* JSON-RPC batches are rejected (-32600). MCP 2025-06-18 removed batching, and
  allowing them would let an ``admin_*`` call hide inside an array.
* ``admin_`` prefix matching is case-insensitive and whitespace-trimmed so that
  ``Admin_reset_key`` / ``" admin_reset_key"`` cannot bypass the check if the
  downstream normalises names.
* Fail closed: anything unparseable is rejected at the gateway, not forwarded.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [mcp-gateway] %(message)s")
log = logging.getLogger("mcp-gateway")

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
UNAUTHORIZED_TOOL_CALL = -32001
UNAUTHENTICATED = -32002  # gateway-specific (server-defined range -32000..-32099)
UPSTREAM_UNAVAILABLE = -32003

VALID_ROLES = {"admin", "viewer"}
ADMIN_TOOL_PREFIX = "admin_"

# Hop-by-hop headers (RFC 9110 §7.6.1) + things the proxy must own.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length", "authorization",
}
# Headers clients may send to the MCP server through the gateway.
FORWARDED_REQUEST_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None, status: int = 200, headers: dict | None = None) -> JSONResponse:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": err}, status_code=status, headers=headers)


class AuthError(Exception):
    pass


def extract_role(authorization: str | None, secret: str) -> tuple[str, str]:
    """Return (subject, role) from 'Bearer <jwt>' or raise AuthError."""
    if not authorization:
        raise AuthError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'")
    try:
        claims = jwt.decode(token.strip(), secret, algorithms=["HS256"], options={"require": ["exp", "sub", "role"]})
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc.__class__.__name__}") from None
    role = claims.get("role")
    if role not in VALID_ROLES:
        raise AuthError("token has an unrecognised role")
    return str(claims["sub"]), role


def is_admin_tool(name: str) -> bool:
    return name.strip().lower().startswith(ADMIN_TOOL_PREFIX)


def authorize(message: dict[str, Any], role: str) -> JSONResponse | None:
    """Return an error response if the message must be blocked, else None (forward)."""
    method = message.get("method")
    if method != "tools/call":
        # tools/list, initialize, ping, notifications, ... -> transparent.
        return None

    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        return jsonrpc_error(message.get("id"), -32602, "Invalid params: tools/call requires string params.name")

    if is_admin_tool(name) and role != "admin":
        return jsonrpc_error(
            message.get("id"),
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            {"tool": name, "required_role": "admin", "role": role},
        )
    return None


def create_app(
    downstream_url: str | None = None,
    jwt_secret: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    upstream_timeout: float = 30.0,
) -> FastAPI:
    downstream_url = downstream_url or os.environ.get("MCP_DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp")
    jwt_secret = jwt_secret or os.environ.get("GATEWAY_JWT_SECRET", "dev-secret-change-me-32-bytes-min!!")
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(upstream_timeout, connect=5.0))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if http_client is None:  # only close clients we created
            await client.aclose()

    app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)
        try:
            sub, role = extract_role(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            log.warning("auth failed: %s", exc)
            return jsonrpc_error(None, UNAUTHENTICATED, "Unauthenticated", str(exc), status=401,
                                 headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'})
        request.state.sub, request.state.role = sub, role
        return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    async def forward(request: Request, body: bytes | None) -> Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARDED_REQUEST_HEADERS}
        headers["x-gateway-user"] = request.state.sub
        headers["x-gateway-role"] = request.state.role
        upstream_req = client.build_request(request.method, downstream_url, headers=headers, content=body,
                                            params=request.query_params)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("downstream unreachable: %r", exc)
            return jsonrpc_error(None, UPSTREAM_UNAVAILABLE, "Downstream MCP server unavailable", status=502)

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != "content-encoding"}
        # Stream through so SSE responses (Streamable HTTP) work without buffering.
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=resp_headers,
                                 background=BackgroundTask(upstream.aclose))

    @app.post("/mcp")
    async def mcp_post(request: Request):
        body = await request.body()
        try:
            message = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return jsonrpc_error(None, PARSE_ERROR, "Parse error")

        if isinstance(message, list):
            return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: JSON-RPC batching is not supported")
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: expected a JSON-RPC 2.0 object")

        # Responses/notifications from the client (no method) just pass through.
        if "method" in message and not isinstance(message["method"], str):
            return jsonrpc_error(message.get("id"), INVALID_REQUEST, "Invalid Request: method must be a string")

        blocked = authorize(message, request.state.role)
        if blocked is not None:
            params = message.get("params")
            log.warning("BLOCKED user=%s role=%s method=%s tool=%r", request.state.sub, request.state.role,
                        message.get("method"), params.get("name") if isinstance(params, dict) else None)
            return blocked

        log.info("forward user=%s role=%s method=%s", request.state.sub, request.state.role, message.get("method"))
        return await forward(request, body)

    # Streamable HTTP also uses GET (server->client SSE) and DELETE (end session).
    @app.api_route("/mcp", methods=["GET", "DELETE"])
    async def mcp_other(request: Request):
        return await forward(request, None)

    return app


def mint_token(sub: str, role: str, secret: str | None = None, ttl_s: int = 3600) -> str:
    """Helper for demos/tests."""
    import time

    secret = secret or os.environ.get("GATEWAY_JWT_SECRET", "dev-secret-change-me-32-bytes-min!!")
    now = int(time.time())
    return jwt.encode({"sub": sub, "role": role, "iat": now, "exp": now + ttl_s}, secret, algorithm="HS256")


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9000")))
