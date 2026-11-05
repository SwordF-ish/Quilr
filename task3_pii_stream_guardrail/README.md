# Task 3: LLM Gateway Streaming Guardrail (PII redaction)

An OpenAI-compatible `POST /v1/chat/completions` endpoint. It proxies requests to an LLM provider and redacts **emails, SSNs and credit-card numbers** to `[REDACTED]` in real time, even when a value is split across stream chunks:

```
upstream deltas:  "...email is jane" | ".doe@exa" | "mple.com and SSN 123-" | "45-67" | "89."
client receives:  "...email is "     | "[REDACTED] and SSN " | "[REDACTED]."
```

## Run

```bash
cd task3_pii_stream_guardrail
python mock_llm_provider.py &      # :9101, streams PII cut at awkward boundaries
python llm_gateway.py &            # :9100
curl -N localhost:9100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"any","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```
To use a real provider, set `UPSTREAM_BASE_URL` and `UPSTREAM_API_KEY`. Any OpenAI-compatible API works (e.g. OpenAI, vLLM, Ollama, LiteLLM).

## Algorithm: `StreamRedactor` (redactor.py)

The class keeps a small **holdback** buffer of text that might still turn into PII once more characters arrive. Everything before it is final, gets redacted, and is emitted right away.

The emit boundary (the *cut*) is placed at the earliest of these points:
1. **The start of the unfinished word.** Email addresses and dashed SSNs contain no whitespace, so a match can't cross a word boundary.
2. **The start of a trailing run of digit groups**, such as `4111 1111 `. Card numbers and spaced SSNs *do* contain spaces, so a digit run is held until a non-digit arrives.
3. **Never inside a candidate match.** If a regex match crosses the cut, the cut moves back to the start of that match's word. Example: in `123 45 6789x`, the unfinished word is `6789x`, but the SSN starts earlier.
4. **Hard cap `max_holdback = 256` chars.** If a single "word" grows past this (e.g. a base64 blob), the redactor flushes anyway. Memory stays **O(256) per stream** however long the response is.

**Correctness guarantee:** for any way of splitting the text into chunks, the streamed output is identical to running `redact()` on the full text. The test suite checks this property against every 2-way split of each PII sample, a char-by-char stream, and 3,000 random adversarial chunkings. A 40k-case fuzz run during development found zero mismatches.

**Latency:** ordinary prose is released as soon as the next whitespace arrives. LLM tokens usually start with a space, so the delay is about one token. The first word reaches the client before the upstream finishes; `test_real_sockets_first_chunk_arrives_long_before_stream_ends` checks this over real TCP.

**Precision**
- Card numbers must be 13–19 digits **and pass the Luhn check**, so order numbers and timestamps are left alone.
- SSNs that can never be issued (area `000`, `666` or `9xx`; group `00`; serial `0000`) are not matched.
- Phone numbers and dates are not matched.

## Stream handling (llm_gateway.py)

- The pipeline is a chain of async generators: `httpx.aiter_lines()` → SSE event framing → JSON parse → per-choice `StreamRedactor` → re-serialized SSE. The full response is never assembled.
- There is one redactor per `choices[i].index`, so `n > 1` works.
- When a `finish_reason` arrives, held text is flushed into that same chunk. If the upstream omits `finish_reason` or `[DONE]`, the leftover text is still flushed before the stream closes, so no text is lost.
- Events whose content is fully held back are skipped. Events carrying `role`, `finish_reason`, tool calls or usage are always forwarded.
- `finally: upstream.aclose()` runs even when the client disconnects, which releases the upstream connection right away.
- Non-streaming requests (`stream: false`) are redacted with the one-shot function.
- Responses set `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so proxies such as nginx don't buffer the stream.

## Limitations / next steps
- `tool_calls[].function.arguments` deltas are not redacted yet. The same `StreamRedactor` could be applied per tool call.
- Detection is regex-based. A production system would add an NER model (e.g. Presidio) running alongside, using the same holdback approach.

## Tests
`pytest task3_pii_stream_guardrail`: 38 tests covering detection, false positives, split-anywhere equivalence, the memory bound, incremental TTFT, and HTTP end-to-end over both ASGI and real sockets.
