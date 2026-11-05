"""Task 1 - Custom MCP server with strict validation over stdio.

Tools
-----
* get_customer_record(customer_id: "CUST-XXXXX")
* trigger_refund(customer_id, amount > 0, reason >= 10 chars)

Design notes
------------
* Uses the low-level ``mcp.server.lowlevel.Server`` so we fully control how
  validation failures are reported: malformed arguments raise ``MCPError`` with
  the standard JSON-RPC code ``-32602 Invalid params``. An unknown tool name is
  also ``-32602`` (not ``-32601 Method not found``): the method ``tools/call``
  exists; it is the ``name`` param that is invalid.
* Business-level failures (e.g. customer not found) are returned as a
  ``CallToolResult`` with ``isError=True`` - this is the MCP convention: the
  protocol call succeeded, the tool reports a domain error the LLM can see.
* STDIO isolation: stdout is reserved for JSON-RPC frames. All logging is
  configured to stderr, there are no print() calls, and ``stdio_server()``
  additionally re-points fd 1 to stderr while serving, so even a stray
  print() from a third-party library cannot corrupt the wire.
"""

from __future__ import annotations

import json
import logging
import sys
from decimal import Decimal
from typing import Annotated, Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Logging -> stderr ONLY
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [customer-mcp] %(message)s",
)
log = logging.getLogger("customer-mcp")

# ---------------------------------------------------------------------------
# Schemas (Pydantic, strict mode)
# ---------------------------------------------------------------------------
CustomerId = Annotated[
    str,
    StringConstraints(pattern=r"^CUST-\d{5}$", strict=True),
    Field(description="Customer identifier formatted as CUST-XXXXX (5 digits)", examples=["CUST-00042"]),
]


class StrictModel(BaseModel):
    # extra="forbid": unknown keys are rejected; strict=True: no "12" -> 12 coercion.
    model_config = ConfigDict(extra="forbid", strict=True)


class GetCustomerRecordInput(StrictModel):
    customer_id: CustomerId


class TriggerRefundInput(StrictModel):
    customer_id: CustomerId
    amount: float = Field(gt=0, le=1_000_000, allow_inf_nan=False, description="Refund amount, positive")
    reason: Annotated[str, StringConstraints(min_length=10, max_length=1000, strip_whitespace=True)]

    @field_validator("amount", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        # bool is a subclass of int in Python - never accept True/False as money.
        if isinstance(v, bool):
            raise ValueError("amount must be a number, not a boolean")
        # Accept JSON integers (e.g. 50) as floats, but nothing else non-numeric.
        if isinstance(v, int):
            return float(v)
        return v

    @field_validator("amount")
    @classmethod
    def _max_two_decimals(cls, v: float) -> float:
        if Decimal(str(v)).as_tuple().exponent < -2:  # type: ignore[operator]
            raise ValueError("amount may have at most 2 decimal places")
        return v


TOOL_MODELS: dict[str, type[StrictModel]] = {
    "get_customer_record": GetCustomerRecordInput,
    "trigger_refund": TriggerRefundInput,
}

TOOLS = [
    types.Tool(
        name="get_customer_record",
        description="Fetch a customer record by ID (format CUST-XXXXX).",
        input_schema=GetCustomerRecordInput.model_json_schema(),
    ),
    types.Tool(
        name="trigger_refund",
        description="Issue a refund to a customer. amount must be > 0; reason must be at least 10 characters.",
        input_schema=TriggerRefundInput.model_json_schema(),
    ),
]

# ---------------------------------------------------------------------------
# Mock data layer
# ---------------------------------------------------------------------------
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "tier": "gold", "balance": 120.50},
    "CUST-00042": {"name": "Alan Turing", "tier": "silver", "balance": 42.00},
    "CUST-12345": {"name": "Grace Hopper", "tier": "platinum", "balance": 999.99},
}
_refund_seq = 0


def _format_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Client-safe validation details (no input echo beyond field path + message)."""
    return [
        {"field": ".".join(str(p) for p in e["loc"]) or "<root>", "type": e["type"], "message": e["msg"]}
        for e in exc.errors(include_url=False, include_input=False)
    ]


def validate_arguments(name: str, arguments: dict[str, Any] | None) -> StrictModel:
    model = TOOL_MODELS.get(name)
    if model is None:
        raise MCPError(types.INVALID_PARAMS, f"Unknown tool: {name}", {"available_tools": list(TOOL_MODELS)})
    if arguments is not None and not isinstance(arguments, dict):
        raise MCPError(types.INVALID_PARAMS, "arguments must be a JSON object")
    try:
        return model.model_validate(arguments or {})
    except ValidationError as exc:
        details = _format_validation_error(exc)
        log.warning("rejected %s call: %s", name, details)
        raise MCPError(types.INVALID_PARAMS, f"Invalid arguments for tool '{name}'", {"errors": details}) from None


def _text(payload: dict[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        is_error=is_error,
    )


def get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _text({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
    return _text({"customer_id": args.customer_id, **record})


def trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    global _refund_seq
    if args.customer_id not in CUSTOMERS:
        return _text({"error": "customer_not_found", "customer_id": args.customer_id}, is_error=True)
    _refund_seq += 1
    log.info("refund issued customer=%s amount=%.2f", args.customer_id, args.amount)
    return _text(
        {
            "refund_id": f"RF-{_refund_seq:06d}",
            "customer_id": args.customer_id,
            "amount": round(args.amount, 2),
            "reason": args.reason,
            "status": "queued",
        }
    )


HANDLERS = {"get_customer_record": get_customer_record, "trigger_refund": trigger_refund}


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------
async def on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    args = validate_arguments(params.name, params.arguments)  # raises MCPError(-32602)
    try:
        return HANDLERS[params.name](args)  # type: ignore[arg-type]
    except MCPError:
        raise
    except Exception:  # never leak tracebacks over the wire
        log.exception("tool %s crashed", params.name)
        raise MCPError(types.INTERNAL_ERROR, "Internal error while executing tool") from None


server = Server(
    "customer-support-mcp",
    version="1.0.0",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


async def main() -> None:
    log.info("starting customer-support-mcp on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
