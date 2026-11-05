"""Mock OpenAI-compatible provider that streams PII deliberately split across chunks."""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_TEXT = (
    "Sure! The customer's email is jane.doe@example.com and their SSN is 123-45-6789. "
    "The card on file is 4111 1111 1111 1111, backup card 5500-0055-5555-5559. "
    "Order 12345678 shipped on 2024-01-05. Reach support at help@acme.io for more."
)

# Chunk boundaries chosen to cut straight through every PII value.
DEFAULT_CHUNKS = ["Sure! The customer's email is jane", ".doe@exa", "mple.com and their SSN is 123-", "45-67", "89. ",
                  "The card on file is 4111 11", "11 1111 1111, backup card 5500-0055-", "5555-5559. ",
                  "Order 12345678 shipped on 2024-01-05. Reach support at help@ac", "me.io for more."]
assert "".join(DEFAULT_CHUNKS) == DEFAULT_TEXT

app = FastAPI(title="Mock LLM provider")
CHUNK_DELAY_S = float(os.environ.get("MOCK_CHUNK_DELAY_S", "0.05"))


def chunk_obj(content: str | None, finish: str | None = None, role: str | None = None) -> dict:
    delta: dict = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": "mock-1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    chunks = body.get("mock_chunks") or DEFAULT_CHUNKS
    if not body.get("stream"):
        return JSONResponse({"id": "chatcmpl-mock", "object": "chat.completion", "model": "mock-1",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(chunks)},
                                          "finish_reason": "stop"}]})

    async def gen():
        yield f"data: {json.dumps(chunk_obj('', role='assistant'))}\n\n"
        for c in chunks:
            await asyncio.sleep(CHUNK_DELAY_S)
            yield f"data: {json.dumps(chunk_obj(c))}\n\n"
        yield f"data: {json.dumps(chunk_obj(None, finish='stop'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9101")))
