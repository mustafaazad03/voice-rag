"""Overlap handling at retrieval time.

Overlapping windows are good for recall and terrible for a top-k list: three of
the five slots end up being the same sentence shifted by one. After fusion we
merge candidates from the same document whose char spans touch or overlap,
keeping the best score and widening the span, so top-k holds k *distinct* facts.
"""

from __future__ import annotations

from ..models import Chunk, ScoredChunk


def merge_overlapping(
    candidates: list[ScoredChunk], *, gap_tolerance: int = 24, max_chars: int = 1400
) -> list[ScoredChunk]:
    """Collapse touching/overlapping spans per document. Order by score, stable."""
    if len(candidates) < 2:
        return candidates

    by_doc: dict[str, list[ScoredChunk]] = {}
    for c in candidates:
        by_doc.setdefault(c.chunk.doc_id, []).append(c)

    merged: list[ScoredChunk] = []
    for group in by_doc.values():
        group.sort(key=lambda c: (c.chunk.start, c.chunk.end))
        current = group[0]
        for nxt in group[1:]:
            overlaps = nxt.chunk.start <= current.chunk.end + gap_tolerance
            fits = (max(nxt.chunk.end, current.chunk.end) - current.chunk.start) <= max_chars
            if overlaps and fits:
                current = _fuse(current, nxt)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)

    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


def _fuse(a: ScoredChunk, b: ScoredChunk) -> ScoredChunk:
    keep, other = (a, b) if a.score >= b.score else (b, a)
    text = _stitch(a.chunk, b.chunk)
    chunk = keep.chunk.model_copy(
        update={
            "text": text,
            "start": min(a.chunk.start, b.chunk.start),
            "end": max(a.chunk.end, b.chunk.end),
            "parent_text": keep.chunk.parent_text or other.chunk.parent_text,
            "meta": {**keep.chunk.meta, "merged_from": [a.chunk.chunk_id, b.chunk.chunk_id]},
        }
    )
    return ScoredChunk(
        chunk=chunk,
        dense_score=max(a.dense_score, b.dense_score),
        sparse_score=max(a.sparse_score, b.sparse_score),
        fusion_score=max(a.fusion_score, b.fusion_score),
        rerank_score=max(a.rerank_score, b.rerank_score),
        dense_rank=_min_rank(a.dense_rank, b.dense_rank),
        sparse_rank=_min_rank(a.sparse_rank, b.sparse_rank),
    )


def _stitch(a: Chunk, b: Chunk) -> str:
    """Join two spans without repeating the overlapping tail."""
    left, right = (a, b) if a.start <= b.start else (b, a)
    if right.end <= left.end:
        return left.text
    # Longest suffix of `left.text` that prefixes `right.text`.
    limit = min(len(left.text), len(right.text))
    for size in range(limit, 5, -1):
        if left.text[-size:] == right.text[:size]:
            return left.text + right.text[size:]
    return f"{left.text} {right.text}".strip()


def _min_rank(a: int | None, b: int | None) -> int | None:
    ranks = [r for r in (a, b) if r is not None]
    return min(ranks) if ranks else None
