"""Mock OpenAI-compatible providers with controllable failure modes.

    /ok/v1/chat/completions         -> 200 with usage
    /ratelimit/v1/chat/completions  -> 429
    /slow/v1/chat/completions       -> sleeps SLOW_S (default 5 s) then 200
    /error/v1/chat/completions      -> 500 with a fake stack trace (must never leak)
    /badrequest/v1/chat/completions -> 400
    /flaky/v1/chat/completions      -> cycles ok -> 429 -> slow (for the live demo)
"""

from __future__ import annotations

import asyncio
import itertools
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="Mock LLM providers")
SLOW_S = float(os.environ.get("MOCK_SLOW_S", "5"))
CALLS: list[str] = []
_flaky = itertools.cycle(["ok", "ratelimit", "slow"])

LEAKY_TRACE = (
    'Traceback (most recent call last):\n  File "/srv/internal/inference/worker.py", line 88, in run\n'
    "    conn = psycopg.connect('postgres://admin:SUPERSECRET@10.0.3.7/models')\nRuntimeError: CUDA OOM on gpu-node-17"
)


def completion(model: str, prompt_chars: int) -> dict:
    prompt_tokens = max(1, prompt_chars // 4)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Hello from {model}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7, "total_tokens": prompt_tokens + 7},
    }


@app.post("/{mode}/v1/chat/completions")
async def chat(mode: str, request: Request):
    body = await request.json()
    if mode == "flaky":
        mode = next(_flaky)
    CALLS.append(mode)
    chars = sum(len(m.get("content") or "") for m in body.get("messages", []) if isinstance(m.get("content"), str))
    if mode == "ok":
        return completion(body.get("model", "mock"), chars)
    if mode == "ratelimit":
        return JSONResponse({"error": {"message": "Rate limit reached for org-internal-123"}}, status_code=429,
                            headers={"retry-after": "20"})
    if mode == "slow":
        await asyncio.sleep(SLOW_S)
        return completion(body.get("model", "mock"), chars)
    if mode == "error":
        return PlainTextResponse(LEAKY_TRACE, status_code=500)
    if mode == "badrequest":
        return JSONResponse({"error": {"message": f"model {body.get('model')} at 10.0.3.7 rejected input"}}, status_code=400)
    return JSONResponse({"error": "unknown mode"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9201")))
