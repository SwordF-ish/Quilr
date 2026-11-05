# Task 4: Rate-Limiting and Model Fallback Router

```
POST /v1/chat/completions  (Bearer <tenant key>)
  │
  ├─ auth: sha256(key) lookup in SQLite ─────────────── 401
  ├─ estimate tokens = prompt≈chars/4 + 4/msg + max_tokens
  ├─ sliding-window reserve (SQLite, BEGIN IMMEDIATE) ── 429 + Retry-After  (413 if > limit)
  ├─ primary  (deadline 3000 ms) ── 429 / timeout / 5xx / conn err ─┐
  │                                                                  ▼
  │                                          secondary ── fail ──► 503 (tokens refunded)
  └─ reconcile reservation → provider-reported usage.total_tokens
```

## Run

```bash
cd task4_llm_router
python mock_providers.py &          # :9201  (/ok, /ratelimit, /slow, /error, /flaky)
python llm_router_app.py &          # :9200, DB at data/gateway.sqlite3, seeds key sk-demo-tenant
for i in 1 2 3; do
  curl -s -i localhost:9200/v1/chat/completions -H 'Authorization: Bearer sk-demo-tenant' \
    -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":50}' \
    | grep -E 'x-gateway-fallback|"model"'
done
# 1st: primary ok   2nd: primary 429 → fallback   3rd: primary hangs → fallback after ~3.0 s
```

Configuration: `GATEWAY_DB_PATH`, `PRIMARY_BASE_URL`, `PRIMARY_API_KEY`, `PRIMARY_MODEL`, `SECONDARY_*` (same three), `DEMO_TENANT_KEY`.

## Rate limiter (`rate_limiter.py`)

**Algorithm.** The limiter is a sliding *log*. Each admitted request writes a row `token_usage(tenant_id, ts, tokens)`. A request is admitted only if `SUM(tokens WHERE ts > now-60) + requested ≤ limit`.
- The window is exact, so there are no bursts at fixed-window boundaries.
- The default limit is 50,000 tokens per minute. Each tenant can have its own limit in the `tenants` table.

**Concurrency.** Eviction, sum, check and insert run in a single `BEGIN IMMEDIATE` transaction.
- That transaction takes SQLite's write lock up front, so two requests can't both see the same remaining budget.
- This holds across threads **and across multiple gateway worker processes** that share the database file.
- WAL mode, `busy_timeout` and one connection per thread keep this fast.
- Blocking SQLite calls run in `asyncio.to_thread`, so they never stall the event loop.
- Test: 200 threads each request 1,000 tokens at the same moment. **Exactly 50** are admitted.

**Eviction**
- Each check deletes that tenant's expired rows inside the same transaction.
- A background task purges rows for idle tenants every 30 s.
- Stored state is therefore bounded by the number of requests in the last 60 s.

**Token tracking**
- Before the call, the gateway reserves an estimate (prompt plus `max_tokens`), which errs on the high side. After the response, it **reconciles** the row to the provider's `usage.total_tokens`.
- If no provider succeeds, or the client disconnects (`CancelledError`), the reservation is **released**. The `finally` block runs the release under `asyncio.shield`, so cancellation can't interrupt it.

**Retry-After** is calculated, not guessed. The limiter walks the window from the oldest row until enough tokens would have expired, and reports how long that takes.

**Persistence.** The state lives on disk, so it survives restarts. Only SHA-256 hashes of API keys are stored.

## Failover (`router.py`)

**When it fails over**

| Primary outcome | Action |
|---|---|
| 429 | fail over |
| no complete response within 3000 ms (connect, headers *and* body) | fail over |
| 5xx, connection error, malformed body | fail over |
| other 4xx | **no** failover (the secondary would reject the same request); sanitized 400 |

**Timeout race.** The primary call runs inside `asyncio.timeout(3.0)`. When the deadline fires, the in-flight request is cancelled and httpx closes the socket. A late primary response can't be read, double-billed, or returned alongside the secondary's: each request has exactly one winner. Tests check that:
- the default deadline fails over at about 3.0 s;
- 30 concurrent timing-out requests each fail over exactly once, and token usage reconciles exactly.

## Error sanitization

Every non-2xx response has exactly this shape:
```json
{"error": {"type": "upstream_error", "code": "all_models_unavailable",
           "message": "The model service is temporarily unavailable. Please retry.",
           "request_id": "req_3f2a..."}}
```

| Source of the error | How it's handled |
|---|---|
| Gateway errors (401, 413, 429) | Raised as `GatewayError` |
| Body validation (400) | Lists the failing field paths only, never the submitted values |
| Starlette HTTP errors (404, …) | Converted to the same envelope |
| Any unexpected exception | A middleware turns it into a generic 500 |

- Upstream response bodies are never forwarded. The mock 500 contains a stack trace, a DB password and internal IPs; tests check that none of that reaches the client.
- Internal model names and hosts are never forwarded either.
- Full details go to stderr, tagged with the `request_id` so logs can be matched to responses.

## Tests
`pytest task4_llm_router`: 32 tests covering window sliding and expiry, Retry-After math, eviction, tenant isolation, restart persistence, thread concurrency, each failover trigger, the real 3 s deadline, leak checks, auth running before validation, and concurrent failover.
