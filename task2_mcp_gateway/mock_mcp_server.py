"""A minimal downstream MCP server (JSON-RPC over HTTP POST) used to exercise the gateway.

It records every request it receives in ``RECEIVED`` so tests can prove that
blocked calls never reached it.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

RECEIVED: list[dict[str, Any]] = []

TOOLS = [
    {"name": "get_weather", "description": "Read-only weather lookup",
     "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "admin_reset_key", "description": "Rotate a tenant's API key (admin only)",
     "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string"}}, "required": ["tenant"]}},
]

app = FastAPI(title="Mock downstream MCP server")


@app.post("/mcp")
async def mcp(request: Request):
    msg = await request.json()
    RECEIVED.append({"msg": msg, "headers": dict(request.headers)})
    method, req_id = msg.get("method"), msg.get("id")

    if req_id is None:  # notification
        return Response(status_code=202)
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "mock-downstream", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments") or {}
        if name == "get_weather":
            text = f"Sunny in {args.get('city', '?')}"
        elif name == "admin_reset_key":
            text = f"Key rotated for {args.get('tenant', '?')}"
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown tool {name}"}})
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    else:
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}})

    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result}, headers={"Mcp-Session-Id": "mock-session-1"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9001")))
