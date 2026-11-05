"""Incremental PII redaction over a text stream.

Problem: PII can be split across stream deltas::

    "Contact jo" | "hn.doe@exa" | "mple.com or 4111 " | "1111 1111 1111"

Naive per-chunk regex misses these; buffering the full response kills TTFT
and memory. ``StreamRedactor`` keeps only a small *holdback* tail:

* Everything up to the start of the current, unfinished "word" (no whitespace
  yet) is final - none of our patterns (email / dashed SSN) contain
  whitespace, so a match can't span that boundary.
* Card numbers and spaced SSNs *do* contain separators, so a trailing run of
  digit groups ("4111 1111 ") is also held until a non-digit arrives.
* The holdback is capped at ``max_holdback`` chars (longer than any PII we
  detect). If a single "word" exceeds it (e.g. a base64 blob), we force-flush
  so memory stays O(max_holdback) per stream regardless of response length.

Result: ordinary prose is released as soon as the next whitespace arrives
(typically the very next token, since LLM tokens usually start with a space),
and the output is identical to redacting the whole text at once.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_EMAIL = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}"
# SSN: AAA-GG-SSSS (dash or single space), excluding never-issued area/group/serial values.
_SSN = r"(?<![\d-])(?!000|666|9\d\d)\d{3}(?P<sep>[- ])(?!00)\d{2}(?P=sep)(?!0000)\d{4}(?![\d-])"
# Card: 13-19 digits, optionally grouped by single spaces/dashes. Luhn-checked below.
_CARD = r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"

PII_RE = re.compile(rf"(?P<email>{_EMAIL})|(?P<card>{_CARD})|(?P<ssn>{_SSN})")

# Trailing digit groups that might still grow into a card/SSN: "4111 1111 " / "123-45-".
_TRAILING_DIGIT_RUN = re.compile(r"\d(?:[ -]?\d)*[ -]?$")
_WORD_TAIL = re.compile(r"\S*$")


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _replace(m: re.Match[str]) -> str:
    if m.lastgroup == "card":
        digits = re.sub(r"\D", "", m.group())
        if not luhn_ok(digits):
            # Not a real card number; it may still contain an SSN (e.g. "123-45-6789 1234").
            return redact(m.group(), _skip_card=True)
    return REDACTED


_SSN_EMAIL_RE = re.compile(rf"(?P<email>{_EMAIL})|(?P<ssn>{_SSN})")


def redact(text: str, *, _skip_card: bool = False) -> str:
    """One-shot redaction (reference implementation)."""
    if _skip_card:
        return _SSN_EMAIL_RE.sub(REDACTED, text)
    return PII_RE.sub(_replace, text)


class StreamRedactor:
    """Stateful, incremental redactor. One instance per output stream / choice."""

    __slots__ = ("_buf", "max_holdback")

    def __init__(self, max_holdback: int = 256) -> None:
        self._buf = ""
        self.max_holdback = max_holdback

    def _safe_cut(self) -> int:
        buf = self._buf
        cut = _WORD_TAIL.search(buf).start()  # start of the unfinished word
        # A trailing digit/separator run can span whitespace (card "4111 1111 ").
        run = _TRAILING_DIGIT_RUN.search(buf)
        if run and run.start() < cut:
            # Hold from the start of the word the run begins in, so the regex
            # lookbehinds see the same left context as a one-shot pass.
            cut = _WORD_TAIL.search(buf, 0, run.start()).start()
        # Never cut through a candidate match (e.g. "123 45 6789x": the SSN
        # spans whitespace but the word "6789x" is the unfinished tail).
        moved = True
        while moved and cut > 0:
            moved = False
            for m in PII_RE.finditer(buf, 0, len(buf)):
                if m.start() < cut < m.end():
                    cut = _WORD_TAIL.search(buf, 0, m.start()).start()
                    moved = True
                    break
                if m.start() >= cut:
                    break
        if len(buf) - cut > self.max_holdback:
            cut = len(buf) - self.max_holdback
        return cut

    def feed(self, delta: str) -> str:
        """Add a delta; return text that is now safe to emit (already redacted)."""
        if not delta:
            return ""
        self._buf += delta
        cut = self._safe_cut()
        if cut <= 0:
            return ""
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return redact(ready)

    def flush(self) -> str:
        """End of stream: emit whatever is held back."""
        out, self._buf = redact(self._buf), ""
        return out

    @property
    def pending(self) -> int:
        return len(self._buf)
