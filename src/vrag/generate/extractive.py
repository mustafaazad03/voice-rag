"""Steps 05 + 06 — constrained generation with citation-backed output.

The 200 ms budget rules out a network LLM call, so the default generator is
extractive: it selects and orders the sentences from the retrieved context that
best answer the query. Every token in the answer provably came from a retrieved
chunk, which makes step 07's hallucination check a verification rather than a
hope. Costs ~2-6 ms.

Selection is MMR-style — relevance to the query minus redundancy against
already-selected sentences — so three near-identical passages do not produce a
three-sentence answer that says the same thing three times.

`generator="llm"` swaps in abstractive phrasing at the cost of the budget; see
generate/llm.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..chunking import normalize_ws, split_sentences
from ..config import Settings, get_settings
from ..models import Citation, ScoredChunk

_TOKEN_RE = re.compile(r"(?u)\w+")
_STOP_SHORT = 2  # tokens this short carry no retrieval signal in any script


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.casefold()) if len(t) > _STOP_SHORT]


def generate(
    query: str, candidates: list[ScoredChunk], settings: Settings | None = None
) -> tuple[str, list[Citation]]:
    s = settings or get_settings()
    # Sort rather than trusting the caller: sentence scoring uses candidate rank as
    # a prior, so an unordered list would quietly produce a differently-ordered
    # answer. The pipeline already hands these over ranked, so this is a no-op there.
    top = sorted(candidates, key=lambda c: c.score, reverse=True)[: s.final_top_k]
    if not top:
        return "", []

    idf = _idf([c.chunk.context for c in top])
    q_tokens = tokens(query)
    q_weight = {t: idf.get(t, math.log(len(top) + 1)) for t in q_tokens}

    pool = _sentence_pool(top, s)
    if not pool:
        return "", []

    for cand in pool:
        cand["score"] = _relevance(cand, q_weight, idf)

    picked = _mmr_select(pool, limit=s.answer_max_sentences)
    picked.sort(key=lambda c: (c["chunk_rank"], c["order"]))

    # Citation markers are assigned in the order they first appear in the answer.
    citations: list[Citation] = []
    marker_by_chunk: dict[str, str] = {}
    parts: list[str] = []
    for cand in picked:
        chunk = cand["chunk"]
        marker = marker_by_chunk.get(chunk.chunk_id)
        if marker is None:
            marker = f"[{len(citations) + 1}]"
            marker_by_chunk[chunk.chunk_id] = marker
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    quote=cand["text"][:280],
                    score=round(cand["chunk_score"], 4),
                )
            )
        parts.append(f"{cand['text']} {marker}")

    answer = " ".join(parts).strip()
    if len(answer) > s.answer_max_chars:
        answer = answer[: s.answer_max_chars].rsplit(" ", 1)[0] + " …"
    return answer, citations


# --------------------------------------------------------------------------- #


def _sentence_pool(top: list[ScoredChunk], s: Settings) -> list[dict]:
    pool: list[dict] = []
    seen: set[str] = set()
    for rank, cand in enumerate(top):
        context = cand.chunk.context
        for order, (start, end) in enumerate(split_sentences(context)):
            text = normalize_ws(context[start:end])
            if len(text) < 20 or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            pool.append(
                {
                    "text": text,
                    "tokens": Counter(tokens(text)),
                    "chunk": cand.chunk,
                    "chunk_score": cand.score,
                    "chunk_rank": rank,
                    "order": order,
                    "score": 0.0,
                }
            )
    return pool


def _relevance(cand: dict, q_weight: dict[str, float], idf: dict[str, float]) -> float:
    counts: Counter = cand["tokens"]
    if not counts:
        return 0.0
    total = sum(counts.values())
    overlap = sum(w for t, w in q_weight.items() if counts[t])
    norm = sum(q_weight.values()) or 1.0
    length_penalty = 1.0 / (1.0 + math.exp((total - 45) / 22.0))  # favour tight sentences
    return (
        0.62 * (overlap / norm)
        + 0.28 * cand["chunk_score"]
        + 0.10 * length_penalty
        - 0.02 * cand["chunk_rank"]
    )


def _mmr_select(pool: list[dict], *, limit: int, lambda_: float = 0.72) -> list[dict]:
    ordered = sorted(pool, key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for cand in ordered:
        if len(picked) >= limit:
            break
        redundancy = max((_jaccard(cand["tokens"], p["tokens"]) for p in picked), default=0.0)
        if lambda_ * cand["score"] - (1 - lambda_) * redundancy <= 0 and picked:
            continue
        if redundancy > 0.62:
            continue
        picked.append(cand)
    return picked or ordered[:1]


def _jaccard(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union else 0.0


def _idf(docs: list[str]) -> dict[str, float]:
    n = len(docs) or 1
    df: Counter = Counter()
    for d in docs:
        df.update(set(tokens(d)))
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
