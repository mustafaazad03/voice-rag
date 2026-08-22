"""Step 06a — input guardrails, run before anything touches the index.

Order matters: normalize, then reject structurally invalid input, then redact PII
(so nothing sensitive reaches logs or a provider), then check safety and
injection. Total cost is under 0.3 ms — regex over a short string.

Off-topic detection is *not* here. You cannot tell whether a question is outside
the corpus without consulting the corpus, so that decision lives in
`retrieve/confidence.py` and runs on the actual retrieval scores.
"""

from __future__ import annotations

import re

from ..config import Settings, get_settings
from ..guardrails import policy
from ..models import GuardResult, GuardVerdict
from ..obs import METRICS

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"(?u)\w")


def check_query(query: str, settings: Settings | None = None) -> GuardResult:
    s = settings or get_settings()
    normalized = _WS_RE.sub(" ", (query or "").strip())

    if len(normalized) < s.min_query_chars or not _WORD_RE.search(normalized):
        return _block("empty_or_gibberish", policy.EMPTY_QUERY, normalized)
    if len(normalized) > s.max_query_chars:
        return _block("too_long", policy.TOO_LONG, normalized[: s.max_query_chars])

    redacted, count = redact(normalized)

    for rule, pattern, message in policy.UNSAFE_RULES:
        if pattern.search(redacted):
            METRICS.inc("guard_block_total", rule=rule)
            return GuardResult(
                verdict=GuardVerdict.BLOCK,
                rule=rule,
                message=message,
                normalized_query=redacted,
                redactions=count,
            )

    if policy.ACTION_REQUEST.search(redacted):
        METRICS.inc("guard_block_total", rule="action_request")
        return GuardResult(
            verdict=GuardVerdict.BLOCK,
            rule="action_request",
            message=policy.ACTION_MESSAGE,
            normalized_query=redacted,
            redactions=count,
        )

    for pattern in policy.INJECTION_PATTERNS:
        if pattern.search(redacted):
            METRICS.inc("guard_block_total", rule="prompt_injection")
            return GuardResult(
                verdict=GuardVerdict.BLOCK,
                rule="prompt_injection",
                message=policy.INJECTION_MESSAGE,
                normalized_query=redacted,
                redactions=count,
            )

    return GuardResult(verdict=GuardVerdict.ALLOW, normalized_query=redacted, redactions=count)


def redact(text: str) -> tuple[str, int]:
    count = 0
    for label, pattern in policy.REDACTIONS:
        text, n = pattern.subn(f"[{label} redacted]", text)
        count += n
    return text, count


def _block(rule: str, message: str, normalized: str) -> GuardResult:
    METRICS.inc("guard_block_total", rule=rule)
    return GuardResult(
        verdict=GuardVerdict.BLOCK, rule=rule, message=message, normalized_query=normalized
    )
