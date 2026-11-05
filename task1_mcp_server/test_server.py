"""Black-box tests: spawn the server as a subprocess and speak raw JSON-RPC over stdio.

This verifies the wire itself (stdout purity + error codes), not just the Python functions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).with_name("server.py")
PROTOCOL_VERSION = "2025-06-18"


class StdioClient:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._id = 0
        self.raw_stdout_lines: list[str] = []

    def send(self, msg: dict) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        assert self.proc.stdout
        line = self.proc.stdout.readline()
        assert line, f"server closed stdout; stderr={self.proc.stderr.read() if self.proc.stderr else ''}"
        self.raw_stdout_lines.append(line)
        return json.loads(line)  # every stdout line MUST be valid JSON

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        while True:
            resp = self.recv()
            if resp.get("id") == self._id:
                return resp

    def call_tool(self, name: str, arguments: dict | None) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> str:
        assert self.proc.stdin
        self.proc.stdin.close()
        try:
            # Drain anything else written to stdout to check purity at shutdown too.
            rest = self.proc.stdout.read() if self.proc.stdout else ""
            self.raw_stdout_lines.extend(l for l in rest.splitlines(keepends=True) if l)
            self.proc.wait(timeout=10)
        finally:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
        return stderr


@pytest.fixture()
def client():
    c = StdioClient()
    init = c.request(
        "initialize",
        {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
    )
    assert "result" in init, init
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield c
    c.close()


def test_tools_list_exposes_both_tools(client):
    resp = client.request("tools/list", {})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {"get_customer_record", "trigger_refund"}
    schema = next(t for t in resp["result"]["tools"] if t["name"] == "trigger_refund")["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}


def test_get_customer_record_ok(client):
    resp = client.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
    result = resp["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["name"] == "Alan Turing"


def test_unknown_customer_is_tool_error_not_protocol_error(client):
    resp = client.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    assert resp["result"]["isError"] is True


def test_trigger_refund_ok_and_int_amount_accepted(client):
    resp = client.call_tool("trigger_refund", {"customer_id": "CUST-00001", "amount": 25, "reason": "Damaged item on arrival"})
    sc = resp["result"]["structuredContent"]
    assert sc["status"] == "queued" and sc["amount"] == 25.0


@pytest.mark.parametrize(
    "tool,args,bad_field",
    [
        ("get_customer_record", {"customer_id": "CUST-1234"}, "customer_id"),        # too short
        ("get_customer_record", {"customer_id": "cust-12345"}, "customer_id"),       # lowercase
        ("get_customer_record", {"customer_id": "CUST-12345 "}, "customer_id"),      # trailing space
        ("get_customer_record", {"customer_id": "CUST-ABCDE"}, "customer_id"),       # not digits
        ("get_customer_record", {"customer_id": 12345}, "customer_id"),              # wrong type
        ("get_customer_record", {}, "customer_id"),                                  # missing
        ("get_customer_record", {"customer_id": "CUST-00001", "x": 1}, "x"),         # extra field
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 0, "reason": "long enough reason"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": -5.0, "reason": "long enough reason"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": "10.0", "reason": "long enough reason"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": True, "reason": "long enough reason"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 1.234, "reason": "long enough reason"}, "amount"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0, "reason": "too short"}, "reason"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0, "reason": "   short    "}, "reason"),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.0}, "reason"),
    ],
)
def test_invalid_inputs_return_jsonrpc_invalid_params(client, tool, args, bad_field):
    resp = client.call_tool(tool, args)
    assert "result" not in resp
    assert resp["error"]["code"] == -32602
    fields = [e["field"] for e in resp["error"]["data"]["errors"]]
    assert any(f.startswith(bad_field) for f in fields), fields


def test_unknown_tool_is_invalid_params(client):
    resp = client.call_tool("drop_database", {})
    assert resp["error"]["code"] == -32602


def test_unknown_method_is_method_not_found(client):
    resp = client.request("does/not/exist", {})
    assert resp["error"]["code"] == -32601


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr():
    c = StdioClient()
    c.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    c.call_tool("trigger_refund", {"customer_id": "CUST-00001", "amount": 10.5, "reason": "Late delivery refund"})
    c.call_tool("get_customer_record", {"customer_id": "bad"})  # triggers a warning log
    stderr = c.close()

    assert c.raw_stdout_lines, "expected JSON-RPC output"
    for line in c.raw_stdout_lines:
        msg = json.loads(line)
        assert msg["jsonrpc"] == "2.0"
    # Logs exist, but only on stderr.
    assert "refund issued" in stderr and "rejected get_customer_record" in stderr
    assert not any("refund issued" in l for l in c.raw_stdout_lines)
