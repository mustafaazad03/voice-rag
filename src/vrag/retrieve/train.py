"""Train the stage-2 reranker on MS MARCO-XI's own relevance labels.

For each held-out query we run stage 1, take the candidates it produced, and
label each one by whether its document was marked `is_selected`. That is exactly
the distribution the reranker sees at serving time — training on gold passages
alone would teach it nothing about the hard negatives stage 1 actually retrieves.
"""

from __future__ import annotations

import numpy as np

from ..config import Settings, get_settings
from ..index.embedder import Embedder
from ..index.store import ChunkStore
from ..ingest.loader import EvalQuery
from ..obs import get_logger
from .hybrid import HybridRetriever
from .rerank import FeatureReranker, build_features, train_reranker

log = get_logger("retrieve.train")


def build_training_set(
    store: ChunkStore,
    embedder: Embedder,
    queries: list[EvalQuery],
    settings: Settings | None = None,
    *,
    max_queries: int = 1500,
    candidates_per_query: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    s = settings or get_settings()
    retriever = HybridRetriever(store, s)

    usable = [q for q in queries if q.relevant_doc_ids][:max_queries]
    if not usable:
        raise ValueError("no queries carry relevance labels; cannot train the reranker")

    vectors = embedder.encode([q.query for q in usable], prefix=s.embed_query_prefix, batch_size=64)

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for q, vec in zip(usable, vectors, strict=True):
        cands = retriever.retrieve(q.query, vec)[:candidates_per_query]
        if not cands:
            continue
        relevant = set(q.relevant_doc_ids)
        y = np.array([1 if c.chunk.doc_id in relevant else 0 for c in cands], dtype=np.int8)
        if y.sum() == 0:  # stage 1 missed entirely: nothing to learn from ordering
            continue
        feats.append(build_features(q.query, cands))
        labels.append(y)

    if not feats:
        raise ValueError("stage 1 retrieved no relevant chunk for any training query")
    X, Y = np.vstack(feats), np.concatenate(labels)
    log.info("training_set_built", rows=len(Y), positives=int(Y.sum()), queries=len(feats))
    return X, Y


def fit(
    store: ChunkStore,
    embedder: Embedder,
    queries: list[EvalQuery],
    settings: Settings | None = None,
    **kw: object,
) -> FeatureReranker:
    X, Y = build_training_set(store, embedder, queries, settings, **kw)
    model = train_reranker(X, Y)
    train_acc = float((model.predict(X) == Y).mean())
    log.info("reranker_trained", train_accuracy=round(train_acc, 4))
    return FeatureReranker(model)
