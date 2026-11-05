from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
import time

import httpx
import pytest
import uvicorn

import mock_llm_provider
from llm_gateway import create_app, redact_chat_stream
from redactor import REDACTED, StreamRedactor, luhn_ok, redact

PII_SAMPLES = [
    "john.doe@example.com",
    "first.last+tag@sub.domain.co.uk",
    "123-45-6789",
    "123 45 6789",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "5500005555555559",
    "378282246310005",  # Amex, 15 digits
]


def stream_through(text: str, sizes) -> str:
    r = StreamRedactor()
    out, i = [], 0
    for n in sizes:
        out.append(r.feed(text[i:i + n]))
        i += n
    out.append(r.feed(text[i:]))
    out.append(r.flush())
    return "".join(out)


# --- one-shot detection --------------------------------------------------------
@pytest.mark.parametrize("pii", PII_SAMPLES)
def test_detects_each_pattern(pii):
    assert redact(f"value: {pii}.") == f"value: {REDACTED}."


@pytest.mark.parametrize(
    "text",
    [
        "Order 12345678 shipped",           # too short for a card
        "1234 5678 9012 3456",               # fails Luhn
        "phone 555-123-4567",                # phone, not SSN
        "invalid ssn 000-12-3456, also 666-12-3456, also 900-12-3456",
        "version 1.2.3 and user@localhost",  # no TLD
        "date 2024-01-05",
    ],
)
def test_no_false_positives(text):
    assert redact(text) == text


def test_luhn():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")


# --- streaming equivalence -------------------------------------------------------
@pytest.mark.parametrize("pii", PII_SAMPLES)
def test_every_two_way_split_is_redacted(pii):
    text = f"Here: {pii} end"
    expected = redact(text)
    assert REDACTED in expected
    for cut in range(1, len(text)):
        assert stream_through(text, [cut]) == expected, cut


@pytest.mark.parametrize("pii", ["john.doe@example.com", "4111 1111 1111 1111", "123-45-6789"])
def test_char_by_char(pii):
    text = f"a {pii} b"
    assert stream_through(text, [1] * len(text)) == redact(text)


def test_randomised_chunking_matches_one_shot():
    rng = random.Random(1234)
    alpha = "0123456789  -@.abX\n"
    for _ in range(3000):
        parts = []
        for _ in range(rng.randint(1, 10)):
            parts.append(rng.choice(PII_SAMPLES) if rng.random() < 0.4
                         else "".join(rng.choice(alpha) for _ in range(rng.randint(1, 10))))
        text = "".join(rng.choice(["", " ", "-", "x"]) + p for p in parts)
        sizes = [rng.randint(1, 7) for _ in range(len(text))]
        assert stream_through(text, sizes) == redact(text), text


# --- latency & memory ------------------------------------------------------------
def test_prose_is_released_immediately_one_word_holdback():
    r = StreamRedactor()
    assert r.feed("Hello") == ""            # unfinished word: held
    assert r.feed(" world") == "Hello "     # released as soon as the next token starts
    assert r.feed(" how are") == "world how "
    assert r.flush() == "are"


def test_memory_bounded_on_unbroken_input():
    r = StreamRedactor(max_holdback=256)
    emitted = 0
    for _ in range(10_000):  # 1 MB with no whitespace at all
        emitted += len(r.feed("x" * 100))
        assert r.pending <= 256
    emitted += len(r.flush())
    assert emitted == 1_000_000


# --- SSE pipeline ------------------------------------------------------------------
def sse_lines(chunks: list[str], finish: bool = True, done: bool = True) -> list[str]:
    lines = []
    for c in chunks:
        lines += ["data: " + json.dumps(mock_llm_provider.chunk_obj(c)), ""]
    if finish:
        lines += ["data: " + json.dumps(mock_llm_provider.chunk_obj(None, finish="stop")), ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


async def collect(lines) -> tuple[str, list[dict], bool]:
    async def src():
        for l in lines:
            yield l

    text, events, saw_done = "", [], False
    async for b in redact_chat_stream(src()):
        payload = b.decode().removeprefix("data: ").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        ev = json.loads(payload)
        events.append(ev)
        for ch in ev["choices"]:
            text += ch["delta"].get("content") or ""
    return text, events, saw_done


async def test_pipeline_redacts_mock_provider_stream():
    text, events, done = await collect(sse_lines(mock_llm_provider.DEFAULT_CHUNKS))
    assert done
    assert text == redact(mock_llm_provider.DEFAULT_TEXT)
    for leak in ["jane", "doe@", "6789", "4111", "5559", "help@"]:
        assert leak not in text
    assert "Order 12345678 shipped" in text
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("finish,done", [(False, True), (False, False)])
async def test_pipeline_flushes_when_upstream_omits_finish_or_done(finish, done):
    text, _, _ = await collect(sse_lines(["mail me: a@b", ".io"], finish=finish, done=done))
    assert text == f"mail me: {REDACTED}"


async def test_pipeline_is_incremental_ttft():
    """The first redacted bytes must leave before the upstream has finished."""
    produced = []

    async def slow_upstream():
        for i, l in enumerate(sse_lines(["Hello", " there", " friend", " bye"])):
            produced.append(i)
            yield l

    gen = redact_chat_stream(slow_upstream())
    first = await gen.__anext__()
    assert b"Hello" in first
    assert len(produced) < len(sse_lines(["Hello", " there", " friend", " bye"]))  # upstream not drained
    await gen.aclose()


# --- HTTP end-to-end ----------------------------------------------------------------
@pytest.fixture()
async def gw():
    mock_llm_provider.CHUNK_DELAY_S = 0
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_llm_provider.app))
    app = create_app(upstream_base_url="http://provider.test/v1", http_client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw.test") as c:
        yield c
    await upstream.aclose()


async def test_http_streaming(gw):
    async with gw.stream("POST", "/v1/chat/completions", json={"model": "m", "stream": True, "messages": []}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = (await r.aread()).decode()
    content = ""
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for ch in json.loads(line[6:])["choices"]:
                content += ch["delta"].get("content") or ""
    assert content == redact(mock_llm_provider.DEFAULT_TEXT)
    assert body.rstrip().endswith("data: [DONE]")


async def test_http_non_streaming(gw):
    r = await gw.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.json()["choices"][0]["message"]["content"] == redact(mock_llm_provider.DEFAULT_TEXT)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_sockets_first_chunk_arrives_long_before_stream_ends():
    """Spin up provider + gateway on real ports; chunks are 150ms apart upstream."""
    mock_llm_provider.CHUNK_DELAY_S = 0.15
    p_port, g_port = _free_port(), _free_port()
    servers = [
        uvicorn.Server(uvicorn.Config(mock_llm_provider.app, port=p_port, log_level="warning")),
        uvicorn.Server(uvicorn.Config(create_app(f"http://127.0.0.1:{p_port}/v1"), port=g_port, log_level="warning")),
    ]
    threads = [threading.Thread(target=s.run, daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        while not all(s.started for s in servers):
            time.sleep(0.02)
        t0 = time.perf_counter()
        first_content_at = None
        with httpx.stream("POST", f"http://127.0.0.1:{g_port}/v1/chat/completions",
                          json={"stream": True, "messages": []}, timeout=10) as r:
            for line in r.iter_lines():
                if line.startswith("data: {") and json.loads(line[6:])["choices"][0]["delta"].get("content"):
                    first_content_at = first_content_at or time.perf_counter() - t0
        total = time.perf_counter() - t0
        n = len(mock_llm_provider.DEFAULT_CHUNKS)
        assert total >= 0.15 * n * 0.9
        assert first_content_at is not None and first_content_at < 0.5, (first_content_at, total)
    finally:
        for s in servers:
            s.should_exit = True
        for t in threads:
            t.join(timeout=5)
        mock_llm_provider.CHUNK_DELAY_S = 0.05
