"""Step 02 — hybrid retrieval: BM25 + embeddings, fused.

Dense retrieval finds paraphrases and cross-lingual matches; BM25 nails rare
tokens, numbers and names that embeddings smear. Running both and fusing beats
either leg alone, and both legs are cheap enough to run unconditionally.

Default fusion is Reciprocal Rank Fusion — rank-based, so it needs no score
calibration between two retrievers with incomparable score scales.
"""

from __future__ import annotations

import numpy as np

from ..config import Settings, get_settings
from ..index.store import ChunkStore
from ..models import ScoredChunk


class HybridRetriever:
    def __init__(self, store: ChunkStore, settings: Settings | None = None) -> None:
        self.store = store
        self._s = settings or get_settings()

    def retrieve(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        fusion: str | None = None,
    ) -> list[ScoredChunk]:
        dense_k = dense_k or self._s.dense_top_k
        sparse_k = sparse_k or self._s.sparse_top_k
        fusion = fusion or self._s.fusion

        d_ids, d_scores = self.store.dense.search(query_vec, dense_k)
        s_ids, s_scores = self.store.sparse.search(query, sparse_k)

        dense = {
            int(i): (rank, float(sc))
            for rank, (i, sc) in enumerate(zip(d_ids, d_scores, strict=True))
        }
        sparse = {
            int(i): (rank, float(sc))
            for rank, (i, sc) in enumerate(zip(s_ids, s_scores, strict=True))
        }

        fuse = _rrf if fusion == "rrf" else _weighted
        fused = fuse(dense, sparse, self._s)

        out: list[ScoredChunk] = []
        for idx, score in fused:
            d = dense.get(idx)
            sp = sparse.get(idx)
            out.append(
                ScoredChunk(
                    chunk=self.store.by_index(idx),
                    dense_score=d[1] if d else 0.0,
                    sparse_score=sp[1] if sp else 0.0,
                    fusion_score=score,
                    dense_rank=d[0] if d else None,
                    sparse_rank=sp[0] if sp else None,
                )
            )
        return out


def _rrf(
    dense: dict[int, tuple[int, float]], sparse: dict[int, tuple[int, float]], s: Settings
) -> list[tuple[int, float]]:
    k = s.rrf_k
    scores: dict[int, float] = {}
    for source in (dense, sparse):
        for idx, (rank, _) in source.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _weighted(
    dense: dict[int, tuple[int, float]], sparse: dict[int, tuple[int, float]], s: Settings
) -> list[tuple[int, float]]:
    dn = _minmax({i: v for i, (_, v) in dense.items()})
    sn = _minmax({i: v for i, (_, v) in sparse.items()})
    w = s.dense_weight
    scores = {idx: w * dn.get(idx, 0.0) + (1 - w) * sn.get(idx, 0.0) for idx in set(dn) | set(sn)}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return dict.fromkeys(values, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}
