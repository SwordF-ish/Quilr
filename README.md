# AI Field Developer – Practice Solutions

Python solutions to the four practice tasks: MCP servers, MCP gateways, LLM gateways, security guardrails and system integration. Every task is runnable, includes mock dependencies (no API keys needed), and has tests.

| # | Task | Folder | Key idea |
|---|------|--------|----------|
| 1 | Custom MCP server with strict validation and stdio transport | [`task1_mcp_server/`](task1_mcp_server) | Strict Pydantic models. Bad input returns JSON-RPC `-32602`. stdout carries only JSON-RPC. |
| 2 | MCP security gateway (tool filtering and auth) | [`task2_mcp_gateway/`](task2_mcp_gateway) | JWT role check. `admin_*` tools blocked with `-32001` before reaching the downstream server. |
| 3 | LLM gateway streaming guardrail (PII redaction) | [`task3_pii_stream_guardrail/`](task3_pii_stream_guardrail) | Holds back only the unfinished word, so memory stays bounded and output matches a one-shot redaction. |
| 4 | Rate-limiting and model fallback router | [`task4_llm_router/`](task4_llm_router) | SQLite sliding-log limiter (`BEGIN IMMEDIATE`). 3000 ms primary deadline. Errors use one sanitized format. |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q            # 124 tests, ~20 s
```

Each task folder has its own README covering the design, run instructions and how it meets the evaluation criteria.

## Layout

```
.
├── task1_mcp_server/            server.py, test_server.py
├── task2_mcp_gateway/           gateway.py, mock_mcp_server.py, test_gateway.py
├── task3_pii_stream_guardrail/  redactor.py, llm_gateway.py, mock_llm_provider.py, test_pii_guardrail.py
├── task4_llm_router/            rate_limiter.py, router.py, llm_router_app.py, mock_providers.py, test_router.py
├── pyproject.toml               pytest config
├── requirements*.txt
└── .github/workflows/tests.yml  CI on Python 3.11 / 3.12
```

## Stack
Python 3.11+, official MCP Python SDK (`mcp` 2.x), Pydantic v2, FastAPI/Starlette, httpx, PyJWT, and SQLite from the standard library.
