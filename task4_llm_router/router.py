"""Primary -> secondary model failover with a hard 3000 ms primary deadline.

Failover triggers (primary only):
  * HTTP 429 Too Many Requests
  * no complete response within ``primary_timeout_s`` (default 3.0 s) - this
    deadline covers connect + headers + body, not just connect
  * transport errors / 5xx (provider is down - the secondary is the better bet)

Not a failover trigger: other 4xx (bad request, auth) - the secondary would
reject the same request, so we surface a sanitised client error instead.

Timeout race: the primary call runs *inside* ``asyncio.timeout``. When the
deadline fires, the in-flight coroutine is cancelled and httpx closes its
connection, so a late primary response can never be read, double-counted or
returned alongside the secondary's. There is exactly one winner per request.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("llm-router")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass
class RouteResult:
    provider: str
    body: dict[str, Any]
    fallback_used: bool
    attempts: list[dict[str, Any]]


class UpstreamClientError(Exception):
    """Provider rejected the request itself (4xx other than 429)."""

    def __init__(self, status: int):
        self.status = status


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[dict[str, Any]], all_rate_limited: bool):
        self.attempts = attempts
        self.all_rate_limited = all_rate_limited


class _Retryable(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class ModelRouter:
    def __init__(
        self,
        primary: ProviderConfig,
        secondary: ProviderConfig,
        client: httpx.AsyncClient,
        primary_timeout_s: float = 3.0,
        secondary_timeout_s: float = 15.0,
    ) -> None:
        self.primary, self.secondary = primary, secondary
        self.client = client
        self.primary_timeout_s = primary_timeout_s
        self.secondary_timeout_s = secondary_timeout_s

    async def _call(self, p: ProviderConfig, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        payload = {**body, "model": p.model}
        try:
            async with asyncio.timeout(timeout_s):
                resp = await self.client.post(
                    f"{p.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {p.api_key}"},
                )
        except TimeoutError:
            raise _Retryable("timeout") from None
        except httpx.HTTPError as exc:
            log.warning("provider=%s transport error: %r", p.name, exc)
            raise _Retryable("transport_error") from None

        if resp.status_code == 429:
            raise _Retryable("rate_limited")
        if resp.status_code >= 500:
            # Log status only; upstream bodies may contain stack traces.
            raise _Retryable(f"upstream_{resp.status_code}")
        if resp.status_code >= 400:
            raise UpstreamClientError(resp.status_code)
        try:
            data = resp.json()
        except ValueError:
            raise _Retryable("invalid_response") from None
        if not isinstance(data, dict) or "choices" not in data:
            raise _Retryable("invalid_response")
        return data

    async def complete(self, body: dict[str, Any]) -> RouteResult:
        attempts: list[dict[str, Any]] = []
        plan = [(self.primary, self.primary_timeout_s), (self.secondary, self.secondary_timeout_s)]
        loop = asyncio.get_running_loop()
        for i, (provider, timeout_s) in enumerate(plan):
            started = loop.time()
            try:
                data = await self._call(provider, body, timeout_s)
            except _Retryable as exc:
                elapsed = round((loop.time() - started) * 1000)
                attempts.append({"provider": provider.name, "outcome": exc.reason, "ms": elapsed})
                log.warning("provider=%s failed (%s) after %sms%s", provider.name, exc.reason, elapsed,
                            " -> failing over" if i == 0 else "")
                continue
            attempts.append({"provider": provider.name, "outcome": "ok", "ms": round((loop.time() - started) * 1000)})
            return RouteResult(provider.name, data, fallback_used=i > 0, attempts=attempts)
        raise AllProvidersFailed(attempts, all_rate_limited=all(a["outcome"] == "rate_limited" for a in attempts))
