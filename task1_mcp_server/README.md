# Task 1: Custom MCP Server (strict validation, stdio)

An MCP server built on the official Python SDK's low-level `Server`. It exposes two tools:

| Tool | Inputs |
|------|--------|
| `get_customer_record` | `customer_id`: `^CUST-\d{5}$` |
| `trigger_refund` | `customer_id`, `amount` (> 0, finite, ≤ 2 decimals), `reason` (≥ 10 chars after trimming) |

## Run

```bash
python task1_mcp_server/server.py          # speaks JSON-RPC on stdin/stdout
npx @modelcontextprotocol/inspector python task1_mcp_server/server.py   # optional GUI
```

Claude Desktop / any MCP client config:
```json
{ "mcpServers": { "customer": { "command": "python", "args": ["/abs/path/task1_mcp_server/server.py"] } } }
```

Manual wire test:
```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"cli","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"CUST-1"}}}' \
 | python task1_mcp_server/server.py 2>/dev/null
# -> ... {"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"Invalid arguments for tool 'get_customer_record'","data":{"errors":[...]}}}
```

## How it meets the criteria

**Stdio isolation**
- The code has no `print()` calls. `logging` is configured with `stream=sys.stderr`.
- The SDK's `stdio_server()` writes JSON-RPC to its own copy of fd 1 and points fd 1 at stderr while serving. A stray `print` from a dependency therefore cannot corrupt the protocol stream.
- `test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr` starts the real process and parses every stdout line as JSON-RPC. It also checks that the log lines appear on stderr.

**Protocol compliance (error mapping)**

| Situation | Response |
|-----------|----------|
| Malformed or missing or extra arguments, wrong types | JSON-RPC error **`-32602` Invalid params**. `data.errors` lists `{field, type, message}` and never echoes the input. |
| Unknown tool name | `-32602` (the method `tools/call` exists; the param is invalid) |
| Unknown JSON-RPC method | `-32601` Method not found (SDK) |
| Unexpected exception inside a tool | `-32603` Internal error with a generic message. The traceback goes to stderr only. |
| Valid input, but the customer does not exist | `CallToolResult` with `isError: true`. This is a domain error the LLM can see, not a protocol error. |

**Validation edge cases.** Pydantic runs in `strict=True` with `extra="forbid"`:
- `"10.0"` is not coerced to `10.0`.
- `True` is not accepted as an amount (Python's `bool` is an `int`).
- NaN and Infinity are rejected, and amounts are capped.
- `"CUST-12345 "` (trailing space) is rejected.
- A reason of 10 characters made up mostly of spaces is rejected, because the text is trimmed before the length check.

The JSON Schema advertised in `tools/list` comes from the same Pydantic models, so the schema clients see always matches what the server enforces.

## Tests
`pytest task1_mcp_server`: 22 black-box tests that drive the real subprocess over raw stdio.
