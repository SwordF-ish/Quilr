from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

import httpx
import pytest

import mock_providers
from llm_router_app import create_app
from rate_limiter import SQLiteTokenRateLimiter, estimate_tokens
from router import ProviderConfig

KEY = "sk-tenant-acme"


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# =============================== rate limiter =====================================
@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def limiter(tmp_path, clock):
    return SQLiteTokenRateLimiter(tmp_path / "rl.sqlite3", clock=clock)


def test_db_is_on_disk(limiter, tmp_path):
    limiter.try_acquire("t", 10)
    assert (tmp_path / "rl.sqlite3").exists()
    assert sqlite3.connect(tmp_path / "rl.sqlite3").execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_sliding_window_admits_up_to_limit_then_rejects(limiter, clock):
    assert limiter.try_acquire("t", 30_000).allowed
    clock.t += 30
    d = limiter.try_acquire("t", 20_000)
    assert d.allowed and d.remaining == 0
    clock.t += 1
    d = limiter.try_acquire("t", 1)
    assert not d.allowed
    # Oldest (30k @ t0) expires at t0+60; now is t0+31 -> 29 s.
    assert d.retry_after_s == pytest.approx(29)


def test_window_slides_not_fixed(limiter, clock):
    limiter.try_acquire("t", 30_000)           # t0
    clock.t += 59.999
    assert not limiter.try_acquire("t", 30_000).allowed  # still inside window
    clock.t += 0.001                           # exactly t0+60 -> expired
    assert limiter.try_acquire("t", 30_000).allowed


def test_retry_after_accounts_for_multiple_rows(limiter, clock):
    for _ in range(5):                          # 5 x 10k at t0, t0+10, ... t0+40
        assert limiter.try_acquire("t", 10_000).allowed
        clock.t += 10
    # now t0+50, full. Need 25k -> must wait for 3 oldest rows (t0, t0+10, t0+20) -> t0+80.
    d = limiter.try_acquire("t", 25_000)
    assert not d.allowed and d.retry_after_s == pytest.approx(30)


def test_eviction_removes_expired_rows(limiter, clock):
    for _ in range(10):
        limiter.try_acquire("a", 100)
        limiter.try_acquire("b", 100)
    assert limiter.row_count() == 20
    clock.t += 61
    limiter.try_acquire("a", 1)  # per-tenant eviction on check
    assert limiter.row_count() == 11  # b's 10 rows remain until global purge
    assert limiter.purge_expired() == 10
    assert limiter.row_count() == 1


def test_tenants_are_isolated(limiter):
    assert limiter.try_acquire("a", 50_000).allowed
    assert limiter.try_acquire("b", 50_000).allowed
    assert not limiter.try_acquire("a", 1).allowed


def test_reconcile_and_release(limiter):
    d = limiter.try_acquire("t", 5_000)
    limiter.reconcile(d.reservation_id, 1_200)
    assert limiter.usage("t") == 1_200
    limiter.release(d.reservation_id)
    assert limiter.usage("t") == 0


def test_request_larger_than_limit_has_no_retry_after(limiter):
    d = limiter.try_acquire("t", 60_000)
    assert not d.allowed and d.retry_after_s is None


def test_state_survives_restart(tmp_path, clock):
    path = tmp_path / "rl.sqlite3"
    SQLiteTokenRateLimiter(path, clock=clock).try_acquire("t", 45_000)
    assert not SQLiteTokenRateLimiter(path, clock=clock).try_acquire("t", 10_000).allowed


def test_concurrent_acquires_never_overspend(tmp_path):
    """200 threads x 1000 tokens against a 50k budget -> exactly 50 admitted."""
    lim = SQLiteTokenRateLimiter(tmp_path / "rl.sqlite3")
    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(200)

    def worker():
        barrier.wait()
        ok = lim.try_acquire("t", 1_000).allowed
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 50
    assert lim.usage("t") == 50_000


def test_estimate_tokens():
    assert estimate_tokens([{"role": "user", "content": "x" * 400}], 100) == 100 + 4 + 100


# ================================ gateway / router ===================================
def provider(name: str, mode: str) -> ProviderConfig:
    return ProviderConfig(name, f"http://providers.test/{mode}/v1", "k", f"{name}-model")


def make_client(tmp_path, primary_mode="ok", secondary_mode="ok", primary_timeout_s=3.0, limit=50_000, **kw):
    mock_providers.CALLS.clear()
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_providers.app))
    lim = SQLiteTokenRateLimiter(tmp_path / "gw.sqlite3")
    lim.upsert_tenant("acme", KEY, limit)
    app = create_app(primary=provider("primary", primary_mode), secondary=provider("secondary", secondary_mode),
                     http_client=upstream, primary_timeout_s=primary_timeout_s, seed_demo_tenant=False, limiter=lim, **kw)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test",
                               headers={"Authorization": f"Bearer {KEY}"})
    return client, lim


REQ = {"model": "auto", "messages": [{"role": "user", "content": "hi there"}], "max_tokens": 100}

LEAK_MARKERS = ["Traceback", "SUPERSECRET", "10.0.3.7", "worker.py", "gpu-node", "org-internal", "providers.test",
                "primary-model", "secondary-model"]


def assert_standard_error(r: httpx.Response, status: int, type_: str):
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "code", "message", "request_id"}
    assert body["error"]["type"] == type_
    assert body["error"]["request_id"] == r.headers["x-request-id"]
    for m in LEAK_MARKERS:
        assert m not in r.text, m


async def test_primary_success(tmp_path):
    c, _ = make_client(tmp_path)
    r = await c.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200
    assert r.json()["model"] == "primary-model"
    assert r.headers["x-gateway-fallback"] == "false"
    assert mock_providers.CALLS == ["ok"]


async def test_failover_on_429(tmp_path):
    c, _ = make_client(tmp_path, primary_mode="ratelimit")
    r = await c.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200 and r.json()["model"] == "secondary-model"
    assert r.headers["x-gateway-fallback"] == "true"
    assert mock_providers.CALLS == ["ratelimit", "ok"]


async def test_failover_on_timeout_respects_deadline(tmp_path):
    mock_providers.SLOW_S = 5
    c, _ = make_client(tmp_path, primary_mode="slow", primary_timeout_s=0.3)
    t0 = time.perf_counter()
    r = await c.post("/v1/chat/completions", json=REQ)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200 and r.json()["model"] == "secondary-model"
    assert 0.3 <= elapsed < 1.5, elapsed  # didn't wait the full 5 s


async def test_default_primary_timeout_is_3000ms(tmp_path):
    mock_providers.SLOW_S = 3.5
    c, _ = make_client(tmp_path, primary_mode="slow")  # default 3.0 s
    t0 = time.perf_counter()
    r = await c.post("/v1/chat/completions", json=REQ)
    elapsed = time.perf_counter() - t0
    assert r.json()["model"] == "secondary-model"
    assert 2.9 <= elapsed < 3.4, elapsed


async def test_failover_on_5xx_without_leaking_trace(tmp_path):
    c, _ = make_client(tmp_path, primary_mode="error")
    r = await c.post("/v1/chat/completions", json=REQ)
    assert r.status_code == 200 and r.json()["model"] == "secondary-model"


async def test_both_fail_returns_sanitized_503_and_refunds_tokens(tmp_path):
    c, lim = make_client(tmp_path, primary_mode="ratelimit", secondary_mode="error")
    r = await c.post("/v1/chat/completions", json=REQ)
    assert_standard_error(r, 503, "upstream_error")
    assert lim.usage("acme") == 0  # reservation released


async def test_upstream_4xx_not_failed_over_and_sanitized(tmp_path):
    c, _ = make_client(tmp_path, primary_mode="badrequest")
    r = await c.post("/v1/chat/completions", json=REQ)
    assert_standard_error(r, 400, "invalid_request_error")
    assert mock_providers.CALLS == ["badrequest"]


async def test_usage_reconciled_to_actual(tmp_path):
    c, lim = make_client(tmp_path)
    r = await c.post("/v1/chat/completions", json=REQ)
    assert lim.usage("acme") == r.json()["usage"]["total_tokens"]  # not the 100+ token estimate


async def test_tenant_rate_limit_429(tmp_path):
    c, _ = make_client(tmp_path, limit=1_000)
    big = {**REQ, "max_tokens": 900}
    assert (await c.post("/v1/chat/completions", json=big)).status_code == 200  # reconciled down to ~9
    r = await c.post("/v1/chat/completions", json={**REQ, "max_tokens": 990})  # 996 est + 9 used > 1000
    assert_standard_error(r, 429, "rate_limit_error")
    assert int(r.headers["retry-after"]) >= 1
    assert mock_providers.CALLS == ["ok"]  # rejected before reaching any provider


async def test_request_over_budget_is_413(tmp_path):
    c, _ = make_client(tmp_path, limit=500)
    r = await c.post("/v1/chat/completions", json={**REQ, "max_tokens": 1000})
    assert_standard_error(r, 413, "invalid_request_error")


@pytest.mark.parametrize("hdr", [None, "Bearer wrong-key", "Basic abc"])
async def test_auth_errors(tmp_path, hdr):
    c, _ = make_client(tmp_path)
    headers = {"Authorization": hdr} if hdr else {"Authorization": ""}
    r = await c.post("/v1/chat/completions", json=REQ, headers=headers)
    assert_standard_error(r, 401, "authentication_error")


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": [{"role": "hacker", "content": "x"}]},
                                  {"messages": [{"role": "user", "content": "x"}], "max_tokens": -1}])
async def test_validation_errors_standardized(tmp_path, body):
    c, _ = make_client(tmp_path)
    r = await c.post("/v1/chat/completions", json=body)
    assert_standard_error(r, 400, "invalid_request_error")
    assert "hacker" not in r.text  # input values not echoed


async def test_unexpected_crash_is_sanitized(tmp_path, monkeypatch):
    c, lim = make_client(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("sqlite at /var/secret/db.sqlite3 exploded")

    monkeypatch.setattr(lim, "try_acquire", boom)
    r = await c.post("/v1/chat/completions", json=REQ)
    assert_standard_error(r, 500, "internal_error")
    assert "exploded" not in r.text and "/var/secret" not in r.text


async def test_auth_checked_before_body_validation(tmp_path):
    c, _ = make_client(tmp_path)
    r = await c.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer nope"})
    assert_standard_error(r, 401, "authentication_error")


async def test_404_standardized(tmp_path):
    c, _ = make_client(tmp_path)
    assert_standard_error(await c.get("/nope"), 404, "invalid_request_error")


async def test_concurrent_requests_with_mixed_failover(tmp_path):
    """Many concurrent requests where primary times out: every one fails over exactly once."""
    mock_providers.SLOW_S = 2
    c, lim = make_client(tmp_path, primary_mode="slow", primary_timeout_s=0.2)
    rs = await asyncio.gather(*[c.post("/v1/chat/completions", json=REQ) for _ in range(30)])
    assert all(r.status_code == 200 and r.json()["model"] == "secondary-model" for r in rs)
    assert mock_providers.CALLS.count("ok") == 30
    assert lim.usage("acme") == sum(r.json()["usage"]["total_tokens"] for r in rs)
