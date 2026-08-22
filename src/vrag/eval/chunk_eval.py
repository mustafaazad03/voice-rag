"""Chunking strategy comparison.

Builds a real index per strategy over the same documents and scores the same
queries against each, so the default strategy is a measurement rather than a
preference. Reports quality (recall/nDCG/MRR), cost (build time, index size,
chunk count) and query latency together — a strategy that wins recall by 2 points
and doubles p95 is not a win at a 200 ms target.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..index.embedder import get_embedder
from ..index.store import ChunkStore
from ..ingest.loader import Corpus, EvalQuery
from ..models import Document
from ..obs import get_logger, percentiles
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import FeatureReranker
from .metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k, summarize

log = get_logger("eval.chunking")

DEFAULT_STRATEGIES = [
    "fixed_char",
    "fixed_token",
    "recursive",
    "sentence_window",
    "semantic",
    "late",
    "hierarchical",
    "metadata_aware",
]


@dataclass
class StrategyReport:
    strategy: str
    chunks: int
    mean_chunk_chars: float
    build_ms: dict
    recall_at_5: float
    recall_at_10: float
    hit_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    query_ms: dict
    error: str | None = None


def evaluate_strategies(
    corpus: Corpus,
    strategies: list[str] | None = None,
    *,
    max_queries: int = 300,
    rerank: bool = True,
) -> list[StrategyReport]:
    s = get_settings()
    embedder = get_embedder()
    reranker = FeatureReranker.load(Path(s.index_dir) / "reranker.joblib") if rerank else None

    queries = [q for q in corpus.queries if q.relevant_doc_ids][:max_queries]
    if not queries:
        raise ValueError("corpus has no labelled queries")
    query_vecs = embedder.encode([q.query for q in queries], prefix=s.embed_query_prefix)

    reports: list[StrategyReport] = []
    for name in strategies or DEFAULT_STRATEGIES:
        try:
            reports.append(
                _evaluate_one(name, corpus.documents, queries, query_vecs, embedder, reranker)
            )
        except Exception as exc:  # noqa: BLE001 - one bad strategy must not stop the sweep
            log.warning("strategy_failed", strategy=name, error=str(exc))
            reports.append(
                StrategyReport(
                    name, 0, 0.0, {}, 0, 0, 0, 0, 0, {}, error=f"{type(exc).__name__}: {exc}"
                )
            )
    return reports


def _evaluate_one(
    strategy: str,
    documents: list[Document],
    queries: list[EvalQuery],
    query_vecs: np.ndarray,
    embedder,
    reranker: FeatureReranker | None,
) -> StrategyReport:
    s = get_settings()
    t0 = time.perf_counter()
    store = ChunkStore.build(documents, strategy=strategy, settings=s, embedder=embedder)
    build_ms = {**store.meta["build_ms"], "total": round((time.perf_counter() - t0) * 1000, 1)}

    retriever = HybridRetriever(store, s)
    rows: list[dict[str, float]] = []
    latencies: list[float] = []

    for q, vec in zip(queries, query_vecs, strict=True):
        t1 = time.perf_counter()
        cands = retriever.retrieve(q.query, vec)
        if reranker is not None:
            head = reranker.score(q.query, cands[: s.rerank_top_n])
            cands = head + cands[s.rerank_top_n :]
        latencies.append((time.perf_counter() - t1) * 1000)

        # De-duplicate to documents: two chunks of the same passage are one hit.
        seen: list[str] = []
        for c in cands:
            if c.chunk.doc_id not in seen:
                seen.append(c.chunk.doc_id)
        relevant = set(q.relevant_doc_ids)
        rows.append(
            {
                "recall_at_5": recall_at_k(seen, relevant, 5),
                "recall_at_10": recall_at_k(seen, relevant, 10),
                "hit_at_5": hit_at_k(seen, relevant, 5),
                "mrr_at_10": mrr_at_k(seen, relevant, 10),
                "ndcg_at_10": ndcg_at_k(seen, relevant, 10),
            }
        )

    agg = summarize(rows)
    return StrategyReport(
        strategy=strategy,
        chunks=store.size,
        mean_chunk_chars=store.meta["chunk_chars"]["mean"],
        build_ms=build_ms,
        query_ms=percentiles(latencies, (50, 95, 100)),
        **agg,
    )


def to_markdown(reports: list[StrategyReport]) -> str:
    header = (
        "| strategy | chunks | mean chars | recall@5 | recall@10 | hit@5 | MRR@10 | nDCG@10 | "
        "retrieve p50 | build s |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in sorted(reports, key=lambda r: -r.ndcg_at_10):
        if r.error:
            lines.append(f"| {r.strategy} | — | — | — | — | — | — | — | — | {r.error} |")
            continue
        lines.append(
            f"| {r.strategy} | {r.chunks} | {r.mean_chunk_chars:.0f} | {r.recall_at_5:.3f} | "
            f"{r.recall_at_10:.3f} | {r.hit_at_5:.3f} | {r.mrr_at_10:.3f} | {r.ndcg_at_10:.3f} | "
            f"{r.query_ms['50']:.2f} ms | {r.build_ms.get('total', 0) / 1000:.1f} |"
        )
    return "\n".join(lines)


def save(reports: list[StrategyReport], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in reports], indent=2))
