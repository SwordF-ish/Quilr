"""Task 3 - LLM Gateway endpoint with a real-time PII-redaction guardrail.

    client ──POST /v1/chat/completions (stream=true)──▶ gateway ──▶ LLM provider (OpenAI-compatible SSE)
           ◀──────────── redacted SSE, chunk by chunk ─────────────┘

Pipeline (all async generators, nothing accumulates the full response):

    upstream bytes ─▶ httpx aiter_lines() ─▶ SSE event framing ─▶ JSON delta
        ─▶ StreamRedactor.feed() per choice ─▶ re-serialise ─▶ client

Memory per stream: one SSE line + O(max_holdback) chars per choice.
Latency: a delta is released as soon as its trailing "word" is complete.
Works against any OpenAI-compatible provider (OpenAI, vLLM, Ollama, LiteLLM, ...)
via UPSTREAM_BASE_URL / UPSTREAM_API_KEY; defaults to the bundled mock provider.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from redactor import StreamRedactor, redact

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [pii-gateway] %(message)s")
log = logging.getLogger("pii-gateway")


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


async def iter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Frame SSE: join ``data:`` lines until a blank line; yield each event's data payload."""
    data_lines: list[str] = []
    async for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
        # ignore comments (":"), event:, id:, retry: - not used by chat-completions streams
    if data_lines:
        yield "\n".join(data_lines)


async def redact_chat_stream(lines: AsyncIterator[str], max_holdback: int = 256) -> AsyncIterator[bytes]:
    """Transform an OpenAI chat.completion.chunk SSE stream, redacting PII in content deltas."""
    redactors: dict[int, StreamRedactor] = {}
    last_chunk: dict[str, Any] | None = None

    def flush_remaining() -> bytes | None:
        # Upstream ended without finish_reason for some choice: emit what's held.
        pending = [(i, r.flush()) for i, r in redactors.items() if r.pending]
        redactors.clear()
        pending = [(i, t) for i, t in pending if t]
        if not pending or last_chunk is None:
            return None
        chunk = {k: v for k, v in last_chunk.items() if k != "choices" and k != "usage"}
        chunk["choices"] = [{"index": i, "delta": {"content": t}, "finish_reason": None} for i, t in pending]
        return _sse(json.dumps(chunk))

    async for data in iter_sse_data(lines):
        if data == "[DONE]":
            tail = flush_remaining()
            if tail:
                yield tail
            yield _sse("[DONE]")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            log.warning("dropping malformed upstream SSE event")
            continue
        last_chunk = chunk

        emit = bool(chunk.get("usage")) or not chunk.get("choices")
        for choice in chunk.get("choices") or []:
            idx = choice.get("index", 0)
            delta = choice.get("delta") or {}
            r = redactors.get(idx)
            if r is None:
                r = redactors[idx] = StreamRedactor(max_holdback)
            content = delta.get("content")
            safe = r.feed(content) if isinstance(content, str) else ""
            if choice.get("finish_reason") is not None:
                safe += r.flush()
            if isinstance(content, str) or safe:
                delta["content"] = safe
                choice["delta"] = delta
            # Keep the event if it carries anything other than an (empty) content delta.
            if safe or choice.get("finish_reason") is not None or any(k != "content" for k in delta):
                emit = True
        if emit:
            yield _sse(json.dumps(chunk))

    tail = flush_remaining()  # upstream closed without [DONE]
    if tail:
        yield tail


def create_app(
    upstream_base_url: str | None = None,
    upstream_api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    base = (upstream_base_url or os.environ.get("UPSTREAM_BASE_URL", "http://127.0.0.1:9101/v1")).rstrip("/")
    api_key = upstream_api_key or os.environ.get("UPSTREAM_API_KEY", "mock-key")
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if http_client is None:
            await client.aclose()

    app = FastAPI(title="LLM Gateway - PII streaming guardrail", lifespan=lifespan)

    def gateway_error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse({"error": {"type": "gateway_error", "code": code, "message": message}}, status_code=status)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return gateway_error(400, "invalid_json", "Request body must be valid JSON")
        if not isinstance(body, dict):
            return gateway_error(400, "invalid_request", "Request body must be a JSON object")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        upstream_req = client.build_request("POST", f"{base}/chat/completions", json=body, headers=headers)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            log.error("upstream unreachable: %r", exc)
            return gateway_error(502, "upstream_unavailable", "LLM provider is unavailable")

        if upstream.status_code != 200:
            await upstream.aread()
            await upstream.aclose()
            log.error("upstream status %s", upstream.status_code)
            return gateway_error(502 if upstream.status_code >= 500 else upstream.status_code,
                                 "upstream_error", "LLM provider returned an error")

        if not body.get("stream"):
            try:
                data = json.loads(await upstream.aread())
            finally:
                await upstream.aclose()
            for choice in data.get("choices", []):
                msg = choice.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = redact(msg["content"])
            return JSONResponse(data)

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for out in redact_chat_stream(upstream.aiter_lines()):
                    yield out
            finally:
                # Runs on normal completion AND on client disconnect (generator cancelled),
                # so the upstream connection is released promptly.
                await upstream.aclose()

        return StreamingResponse(
            body_iter(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9100")))
