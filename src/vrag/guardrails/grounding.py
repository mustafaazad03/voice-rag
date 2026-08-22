"""Step 07 — hallucination check and insufficient-evidence fallback.

Verifies the answer against the evidence that was actually retrieved, per
sentence. The measure is IDF-weighted *containment*: what fraction of a
sentence's informative tokens appear in some single retrieved chunk. Requiring
one chunk to cover the sentence (rather than the union of all chunks) is what
catches the classic failure where a model stitches half a fact from passage A
onto half a fact from passage B.

Lexical rather than an NLI model on purpose — an NLI cross-encoder costs more
than the entire latency budget. On the extractive path this is a true
verification (sentences are copied, so containment is ~1.0) and it is what makes
an LLM-generated answer safe to serve.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..chunking import normalize_ws, split_sentences
from ..config import Settings, get_settings
from ..models import GroundingReport, ScoredChunk

_TOKEN_RE = re.compile(r"(?u)\w+")
_MARKER_RE = re.compile(r"\[\d+\]")


def check_grounding(
    answer: str, candidates: list[ScoredChunk], settings: Settings | None = None
) -> GroundingReport:
    s = settings or get_settings()
    text = _MARKER_RE.sub(" ", answer or "").strip()
    if not text:
        return GroundingReport(grounded=False, score=0.0, unsupported_sentences=[])
    if not candidates:
        return GroundingReport(grounded=False, score=0.0, unsupported_sentences=[text])

    evidence = [Counter(_tokens(c.chunk.context)) for c in candidates]
    idf = _idf(evidence)

    sentences = [normalize_ws(text[a:b]) for a, b in split_sentences(text)] or [text]

    scores: list[float] = []
    unsupported: list[str] = []
    for sentence in sentences:
        toks = _tokens(sentence)
        if not toks:
            continue
        weights = {t: idf.get(t, _MAX_IDF) for t in set(toks)}
        total = sum(weights.values()) or 1.0
        best = max(
            (sum(w for t, w in weights.items() if ev[t]) / total for ev in evidence), default=0.0
        )
        scores.append(best)
        if best < s.min_answer_grounding:
            unsupported.append(sentence)

    overall = min(scores) if scores else 0.0  # the weakest sentence decides
    return GroundingReport(
        grounded=not unsupported and overall >= s.min_answer_grounding,
        score=round(overall, 4),
        unsupported_sentences=unsupported,
    )


_MAX_IDF = 3.0


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.casefold()) if len(t) > 2]


def _idf(evidence: list[Counter]) -> dict[str, float]:
    n = len(evidence) or 1
    df: Counter = Counter()
    for ev in evidence:
        df.update(ev.keys())
    return {t: min(_MAX_IDF, math.log((n + 1) / (c + 0.5))) for t, c in df.items()}
