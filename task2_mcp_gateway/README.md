# Task 2: MCP Security Gateway (tool filtering and auth)

```
agent ──POST /mcp, Bearer <JWT>──▶ gateway ──▶ downstream MCP server
                                     │
                                     ├─ 401  bad/missing/expired token (downstream never called)
                                     ├─ tools/list  → forwarded transparently
                                     ├─ tools/call admin_* & role≠admin → -32001 Unauthorized Tool Call
                                     └─ everything else → forwarded, response streamed back
```

## Run

```bash
cd task2_mcp_gateway
python mock_mcp_server.py &                     # downstream on :9001
python gateway.py &                             # gateway on :9000
VIEWER=$(python -c "from gateway import mint_token; print(mint_token('alice','viewer'))")
ADMIN=$(python -c "from gateway import mint_token; print(mint_token('bob','admin'))")

curl -s localhost:9000/mcp -H "Authorization: Bearer $VIEWER" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
curl -s localhost:9000/mcp -H "Authorization: Bearer $VIEWER" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant":"acme"}}}'
# {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call","data":{...}}}
curl -s localhost:9000/mcp -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant":"acme"}}}'
```

Configuration: `MCP_DOWNSTREAM_URL`, `GATEWAY_JWT_SECRET`.

## Design notes

**Parsing the JSON-RPC wire format**
- Invalid JSON gets `-32700`.
- Anything that isn't a JSON-RPC 2.0 object gets `-32600`, and so does a non-string `method`.
- `tools/call` with a missing or non-string `params.name` gets `-32602`.
- The request `id` is copied into every error the gateway generates.
- **Batches are rejected (`-32600`).** MCP 2025-06-18 removed JSON-RPC batching. Accepting batches would let a request hide `admin_*` calls inside an array.
- JSON-RPC errors return HTTP 200 (the JSON-RPC convention). Authentication failures return HTTP 401 with `WWW-Authenticate`.

**Authorization**
- Tokens are HS256 JWTs. The gateway checks the signature and requires `exp`, `sub` and `role`. Only HS256 is accepted, so `alg: none` tokens are rejected.
- The role must be `admin` or `viewer`. Any other value is rejected (fail closed).
- The `admin_` prefix check is **case-insensitive and ignores surrounding whitespace**. `Admin_Reset_Key` and `" admin_reset_key"` cannot get past a downstream server that normalizes names.
- A name such as `get_admin_stats` does *not* match. Only the prefix counts.

**Proxying**
- The gateway uses a shared `httpx.AsyncClient` with keep-alive.
- The downstream response is **streamed** back (`aiter_raw`), so Streamable-HTTP SSE responses keep working.
- `GET` and `DELETE /mcp` (the SSE channel and session end) are also proxied.
- Hop-by-hop headers are stripped. Only MCP headers are forwarded: `Mcp-Session-Id`, `MCP-Protocol-Version`, `Accept`, `Last-Event-ID`.
- **The client's bearer token is not forwarded downstream.** MCP security guidance warns against passing tokens through. The gateway sends `X-Gateway-User` and `X-Gateway-Role` instead.
- If the downstream server can't be reached, the gateway returns HTTP 502 with error `-32003`.

## Tests
`pytest task2_mcp_gateway`: 32 tests. They cover role checks, prefix-bypass attempts, forged, expired and wrong-signature tokens, batch smuggling and malformed envelopes. They also check that **blocked calls never reach the downstream server**: the mock records every request it receives.
