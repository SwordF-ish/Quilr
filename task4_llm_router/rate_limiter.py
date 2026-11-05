"""Token-aware sliding-window rate limiter persisted in an on-disk SQLite database.

Algorithm: *sliding log*. Every admitted request writes a row
``(tenant_id, ts, tokens)``. A request asking for ``n`` tokens is admitted iff

    SUM(tokens WHERE ts > now - window) + n <= limit

This is exact (no fixed-window boundary bursts) and the state is bounded:
expired rows are evicted on every check for that tenant, plus a periodic
global purge.

Concurrency: check + insert run inside ``BEGIN IMMEDIATE``, which takes
SQLite's RESERVED (write) lock up front. Two concurrent requests - from
different threads *or different gateway worker processes* sharing the file -
therefore cannot both read the same "remaining budget" and overspend it.
WAL mode keeps readers non-blocking; ``busy_timeout`` queues writers.

Token accounting: callers *reserve* an estimate (prompt estimate + max_tokens)
before calling the model, then ``reconcile`` to the provider-reported usage,
or ``release`` the reservation if no provider produced a response.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOKENS_PER_MINUTE = 50_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    api_key_hash      TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL UNIQUE,
    tokens_per_minute INTEGER NOT NULL DEFAULT 50000 CHECK (tokens_per_minute > 0)
);
CREATE TABLE IF NOT EXISTS token_usage (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT    NOT NULL,
    ts        REAL    NOT NULL,
    tokens    INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS ix_token_usage_tenant_ts ON token_usage (tenant_id, ts);
"""


def hash_key(api_key: str) -> str:
    # Raw API keys are never stored - only their SHA-256.
    return hashlib.sha256(api_key.encode()).hexdigest()


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    tokens_per_minute: int


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    used: int                 # tokens in window *before* this request
    requested: int
    reservation_id: int | None = None
    retry_after_s: float | None = None

    @property
    def remaining(self) -> int:
        spent = self.used + (self.requested if self.allowed else 0)
        return max(0, self.limit - spent)


class SQLiteTokenRateLimiter:
    def __init__(self, db_path: str | Path, window_s: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        self.db_path = str(db_path)
        self.window_s = window_s
        self.clock = clock
        self._local = threading.local()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    # One connection per thread (sqlite3 connections must not be shared across threads).
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    # --- tenants -------------------------------------------------------------
    def upsert_tenant(self, tenant_id: str, api_key: str, tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE) -> None:
        self._conn().execute(
            "INSERT INTO tenants (api_key_hash, tenant_id, tokens_per_minute) VALUES (?, ?, ?) "
            "ON CONFLICT(api_key_hash) DO UPDATE SET tenant_id=excluded.tenant_id, tokens_per_minute=excluded.tokens_per_minute",
            (hash_key(api_key), tenant_id, tokens_per_minute),
        )

    def lookup_tenant(self, api_key: str) -> Tenant | None:
        row = self._conn().execute(
            "SELECT tenant_id, tokens_per_minute FROM tenants WHERE api_key_hash = ?", (hash_key(api_key),)
        ).fetchone()
        return Tenant(*row) if row else None

    def tenant_count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM tenants").fetchone()[0]

    # --- limiting --------------------------------------------------------------
    def try_acquire(self, tenant_id: str, tokens: int, limit: int = DEFAULT_TOKENS_PER_MINUTE) -> Decision:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = self.clock()
            cutoff = now - self.window_s
            # Eviction: rows at/older than the window edge no longer count.
            conn.execute("DELETE FROM token_usage WHERE tenant_id = ? AND ts <= ?", (tenant_id, cutoff))
            used = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()[0]

            if used + tokens <= limit:
                cur = conn.execute("INSERT INTO token_usage (tenant_id, ts, tokens) VALUES (?, ?, ?)", (tenant_id, now, tokens))
                conn.execute("COMMIT")
                return Decision(True, limit, used, tokens, reservation_id=cur.lastrowid)

            retry = None
            if tokens <= limit:
                # Walk oldest-first until enough budget would have expired.
                need = used + tokens - limit
                freed = 0
                for ts, t in conn.execute(
                    "SELECT ts, tokens FROM token_usage WHERE tenant_id = ? ORDER BY ts", (tenant_id,)
                ):
                    freed += t
                    if freed >= need:
                        retry = max(0.0, ts + self.window_s - now)
                        break
            conn.execute("COMMIT")  # keep the eviction
            return Decision(False, limit, used, tokens, retry_after_s=retry)
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        """Replace the reserved estimate with the provider-reported usage."""
        self._conn().execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), reservation_id))

    def release(self, reservation_id: int) -> None:
        """Refund a reservation (no model produced a completion)."""
        self._conn().execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def usage(self, tenant_id: str) -> int:
        cutoff = self.clock() - self.window_s
        return self._conn().execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND ts > ?", (tenant_id, cutoff)
        ).fetchone()[0]

    def purge_expired(self) -> int:
        """Global eviction for tenants that went idle (called periodically)."""
        cur = self._conn().execute("DELETE FROM token_usage WHERE ts <= ?", (self.clock() - self.window_s,))
        return cur.rowcount

    def row_count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    """Cheap, conservative pre-flight estimate (~4 chars/token + per-message overhead + completion budget)."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):  # multi-part content
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return math.ceil(chars / 4) + 4 * len(messages) + max_tokens
