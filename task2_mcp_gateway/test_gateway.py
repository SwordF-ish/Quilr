from __future__ import annotations

import time

import httpx
import jwt
import pytest

import mock_mcp_server
from gateway import create_app, mint_token

SECRET = "test-secret-that-is-long-enough-for-hs256"
DOWNSTREAM = "http://downstream.test/mcp"


@pytest.fixture()
async def gw():
    mock_mcp_server.RECEIVED.clear()
    downstream_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_mcp_server.app))
    app = create_app(downstream_url=DOWNSTREAM, jwt_secret=SECRET, http_client=downstream_client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test") as c:
        yield c
    await downstream_client.aclose()


def auth(role: str, **kw) -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token('user-1', role, SECRET, **kw)}"}


def rpc(method: str, params: dict | None = None, id_: int = 1) -> dict:
    m = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        m["params"] = params
    return m


# --- tools/list ---------------------------------------------------------------
@pytest.mark.parametrize("role", ["viewer", "admin"])
async def test_tools_list_forwarded_transparently(gw, role):
    r = await gw.post("/mcp", json=rpc("tools/list"), headers=auth(role))
    assert r.status_code == 200
    assert [t["name"] for t in r.json()["result"]["tools"]] == ["get_weather", "admin_reset_key"]
    assert r.headers["mcp-session-id"] == "mock-session-1"  # response headers proxied
    assert len(mock_mcp_server.RECEIVED) == 1


# --- tools/call authorization ----------------------------------------------------
async def test_viewer_can_call_normal_tool(gw):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "get_weather", "arguments": {"city": "Pune"}}), headers=auth("viewer"))
    assert r.json()["result"]["content"][0]["text"] == "Sunny in Pune"


async def test_admin_can_call_admin_tool(gw):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant": "acme"}}), headers=auth("admin"))
    assert r.json()["result"]["content"][0]["text"] == "Key rotated for acme"


@pytest.mark.parametrize("tool", ["admin_reset_key", "Admin_Reset_Key", "  admin_reset_key", "ADMIN_anything"])
async def test_viewer_blocked_from_admin_tool_and_downstream_not_called(gw, tool):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": tool}, id_=42), headers=auth("viewer"))
    body = r.json()
    assert r.status_code == 200  # JSON-RPC errors travel in a 200 body
    assert body == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {"code": -32001, "message": "Unauthorized Tool Call",
                  "data": {"tool": tool, "required_role": "admin", "role": "viewer"}},
    }
    assert mock_mcp_server.RECEIVED == []  # never reached downstream


async def test_tool_named_like_admin_but_not_prefixed_is_allowed(gw):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "get_admin_stats"}), headers=auth("viewer"))
    # Forwarded (downstream says unknown tool) - gateway did not block it.
    assert r.json()["error"]["code"] == -32602
    assert len(mock_mcp_server.RECEIVED) == 1


@pytest.mark.parametrize("params", [None, {}, {"name": 123}, {"name": ""}, "admin_reset_key"])
async def test_malformed_tools_call_rejected(gw, params):
    msg = rpc("tools/call", params) if params is not None else rpc("tools/call")
    r = await gw.post("/mcp", json=msg, headers=auth("admin"))
    assert r.json()["error"]["code"] == -32602
    assert mock_mcp_server.RECEIVED == []


# --- token handling ----------------------------------------------------------------
async def test_missing_token_401(gw):
    r = await gw.post("/mcp", json=rpc("tools/list"))
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
    assert mock_mcp_server.RECEIVED == []


@pytest.mark.parametrize(
    "header",
    [
        "Basic abc",
        "Bearer ",
        "Bearer not-a-jwt",
        "Bearer " + jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) + 60}, "WRONG-secret-xxxxxxxxxxxxxxxxxxxxxx", algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "role": "superuser", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": "x", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256"),
    ],
    ids=["basic", "empty", "garbage", "bad-signature", "expired", "unknown-role", "no-role"],
)
async def test_invalid_tokens_rejected(gw, header):
    r = await gw.post("/mcp", json=rpc("tools/call", {"name": "admin_reset_key"}), headers={"Authorization": header})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32002
    assert mock_mcp_server.RECEIVED == []


async def test_alg_none_token_rejected(gw):
    forged = jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) + 60}, None, algorithm="none")
    r = await gw.post("/mcp", json=rpc("tools/list"), headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


async def test_client_token_not_forwarded_downstream(gw):
    await gw.post("/mcp", json=rpc("tools/list"), headers=auth("viewer"))
    fwd_headers = mock_mcp_server.RECEIVED[0]["headers"]
    assert "authorization" not in fwd_headers
    assert fwd_headers["x-gateway-role"] == "viewer"


# --- wire format -------------------------------------------------------------------
async def test_parse_error(gw):
    r = await gw.post("/mcp", content=b"{not json", headers={**auth("admin"), "content-type": "application/json"})
    assert r.json()["error"]["code"] == -32700


async def test_batch_rejected_so_admin_calls_cannot_be_smuggled(gw):
    batch = [rpc("tools/list", id_=1), rpc("tools/call", {"name": "admin_reset_key"}, id_=2)]
    r = await gw.post("/mcp", json=batch, headers=auth("viewer"))
    assert r.json()["error"]["code"] == -32600
    assert mock_mcp_server.RECEIVED == []


@pytest.mark.parametrize("payload", [{"id": 1, "method": "tools/list"}, {"jsonrpc": "1.0", "id": 1, "method": "x"}, "str", 5])
async def test_invalid_envelope(gw, payload):
    r = await gw.post("/mcp", json=payload, headers=auth("admin"))
    assert r.json()["error"]["code"] == -32600


async def test_notification_forwarded(gw):
    r = await gw.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=auth("viewer"))
    assert r.status_code == 202


async def test_downstream_down_returns_502():
    app = create_app(downstream_url="http://127.0.0.1:1/mcp", jwt_secret=SECRET)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test") as c:
        r = await c.post("/mcp", json=rpc("tools/list"), headers=auth("viewer"))
    assert r.status_code == 502 and r.json()["error"]["code"] == -32003
