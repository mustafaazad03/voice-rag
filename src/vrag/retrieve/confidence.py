"""Step 04 — source confidence scoring, and the decision to abstain.

Five signals, each catching a different failure:

  top_score         is anything actually relevant?  (raw cosine, not a rank score)
  score_gap         is the winner clearly better than #2, or is this ambiguous?
  agreement         do several independent documents say it, or just one?
  lexical_coverage  are the query's content words present, or only vibes?
  trust             how much do we trust these sources (human-judged > unjudged)?

`top_score` deliberately uses the encoder's cosine similarity rather than the
reranker's output. The reranker emits a calibrated *probability of being the best
candidate in this list*, which is high even when every candidate is garbage — a
threshold on it would never fire. Cosine is comparable across queries, so a
threshold on it means something.

Two outcomes are distinguished, because they deserve different answers:
  off_topic     the corpus does not cover this subject at all
  insufficient  the subject is covered but the evidence is too thin

The diagram's "freshness" input has no counterpart in MS MARCO-XI — the corpus
carries no timestamps — so it is deliberately absent rather than faked.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from ..config import Settings, get_settings
from ..models import ConfidenceReport, ScoredChunk

_TOKEN_RE = re.compile(r"(?u)\w+")
# Weights sum to 1.0. Retrieval strength dominates; the rest are corroboration.
WEIGHTS = {"top": 0.42, "gap": 0.12, "agreement": 0.14, "coverage": 0.22, "trust": 0.10}


def score_confidence(
    query: str, candidates: list[ScoredChunk], settings: Settings | None = None
) -> ConfidenceReport:
    s = settings or get_settings()
    if not candidates:
        return empty_confidence(["no_candidates"], off_topic=True)

    top = candidates[: max(s.final_top_k, 1)]
    # Cosine, not rerank probability. A sparse-only hit contributes 0, so take the
    # best dense evidence available in the top-k rather than position 1's.
    similarities = [c.dense_score for c in top]
    top_score = max(similarities)
    mean_top_k = sum(similarities) / len(similarities)
    ordered = sorted(similarities, reverse=True)
    gap = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    agreement = len({c.chunk.doc_id for c in top}) / len(top)
    coverage = lexical_coverage(query, top)
    trust = sum(float(c.chunk.meta.get("trust", 0.5)) for c in top) / len(top)

    overall = (
        WEIGHTS["top"] * _clip(top_score)
        + WEIGHTS["gap"] * _clip(gap * 3.0)
        + WEIGHTS["agreement"] * agreement
        + WEIGHTS["coverage"] * coverage
        + WEIGHTS["trust"] * trust
    )

    off_topic = top_score < s.off_topic_similarity

    # Lexical coverage only means something when the query and the evidence share a
    # writing system. An Assamese question answered from an English passage scores
    # 0.0 coverage by construction and is still a perfectly good retrieval, so a
    # cross-script match is judged on similarity alone. Very strong similarity is
    # also allowed to stand in for overlap within a script.
    cross_script = dominant_script(query) != dominant_script(top[0].chunk.context)
    covered = cross_script or coverage >= s.min_lexical_coverage or top_score >= s.strong_similarity

    reasons: list[str] = []
    if off_topic:
        reasons.append("off_topic")
    elif top_score < s.min_retrieval_score:
        reasons.append("weak_top_score")
    if cross_script:
        reasons.append("cross_script_match")
    if not covered:
        reasons.append("low_lexical_coverage")
    if agreement < 0.4:
        reasons.append("single_source")

    sufficient = (
        not off_topic
        and top_score >= s.min_retrieval_score
        and covered
        and overall >= s.abstain_confidence
    )

    return ConfidenceReport(
        top_score=round(top_score, 4),
        mean_top_k=round(mean_top_k, 4),
        score_gap=round(gap, 4),
        agreement=round(agreement, 4),
        lexical_coverage=round(coverage, 4),
        trust=round(trust, 4),
        overall=round(overall, 4),
        sufficient=sufficient,
        off_topic=off_topic,
        reasons=reasons,
    )


def lexical_coverage(query: str, candidates: list[ScoredChunk]) -> float:
    """Share of the query's content tokens that appear somewhere in the evidence."""
    q = {t for t in _TOKEN_RE.findall(query.casefold()) if len(t) > 2}
    if not q:
        return 1.0  # nothing to cover, e.g. a very short query in an abugida
    seen: set[str] = set()
    for c in candidates:
        seen |= set(_TOKEN_RE.findall(c.chunk.context.casefold()))
    return len(q & seen) / len(q)


@lru_cache(maxsize=4096)
def _script_of(ch: str) -> str:
    """'क' -> 'DEVANAGARI', 'a' -> 'LATIN'. Unicode names start with the script."""
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return ""


def dominant_script(text: str, sample: int = 200) -> str:
    counts: dict[str, int] = {}
    for ch in text[:sample]:
        if ch.isalpha():
            name = _script_of(ch)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else ""


def empty_confidence(reasons: list[str], *, off_topic: bool = False) -> ConfidenceReport:
    return ConfidenceReport(
        top_score=0.0,
        mean_top_k=0.0,
        score_gap=0.0,
        agreement=0.0,
        lexical_coverage=0.0,
        trust=0.0,
        overall=0.0,
        sufficient=False,
        off_topic=off_topic,
        reasons=reasons,
    )


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))
