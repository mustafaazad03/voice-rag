"""Step 03, second stage — reranking.

A cross-encoder is the textbook answer and costs 50-300 ms on CPU, which spends
the entire latency budget on one stage. Instead this is a *learned feature*
reranker: cheap signals the first stage already computed (both scores, both
ranks, lexical coverage, chunk geometry) fed to a small gradient-boosted model
trained on MS MARCO-XI's own `is_selected` labels. Scoring 24 candidates costs
~0.2 ms and still fixes the ordering mistakes RRF makes.

Falls back to a fixed linear blend when no model has been trained, so retrieval
never hard-depends on the training step.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..models import ScoredChunk
from ..obs import get_logger

log = get_logger("retrieve.rerank")

_TOKEN_RE = re.compile(r"(?u)\w+")
FEATURE_NAMES = [
    "dense_score",
    "sparse_norm",
    "fusion_norm",
    "dense_rr",
    "sparse_rr",
    "in_both",
    "coverage",
    "bigram_hits",
    "log_len",
    "rel_position",
    "log_doc_chunks",
    "score_margin",
]
# Hand-set weights used until a model is trained. Sign and rough magnitude come
# from what each signal means, not from tuning; the trained model replaces them.
HEURISTIC_WEIGHTS = np.array(
    [1.20, 0.55, 0.90, 0.45, 0.30, 0.25, 0.85, 0.35, -0.05, -0.10, -0.05, 0.30]
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def build_features(query: str, candidates: list[ScoredChunk]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    q_tokens = tokenize(query)
    q_set = set(q_tokens)
    q_bigrams = {f"{a} {b}" for a, b in zip(q_tokens, q_tokens[1:], strict=False)}

    sparse = np.array([c.sparse_score for c in candidates], dtype=np.float32)
    fusion = np.array([c.fusion_score for c in candidates], dtype=np.float32)
    sparse_n, fusion_n = _minmax(sparse), _minmax(fusion)
    best_fusion = float(fusion.max()) if len(fusion) else 0.0

    rows = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    for i, cand in enumerate(candidates):
        tokens = tokenize(cand.chunk.text)
        token_set = set(tokens)
        bigrams = {f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)}
        coverage = len(q_set & token_set) / max(len(q_set), 1)
        rows[i] = (
            cand.dense_score,
            sparse_n[i],
            fusion_n[i],
            1.0 / (1.0 + (cand.dense_rank if cand.dense_rank is not None else 99)),
            1.0 / (1.0 + (cand.sparse_rank if cand.sparse_rank is not None else 99)),
            float(cand.dense_rank is not None and cand.sparse_rank is not None),
            coverage,
            min(len(q_bigrams & bigrams), 5) / 5.0,
            math.log1p(len(cand.chunk.text)) / 8.0,
            cand.chunk.position / max(cand.chunk.n_chunks_in_doc, 1),
            math.log1p(cand.chunk.n_chunks_in_doc) / 5.0,
            (cand.fusion_score - best_fusion),
        )
    return rows


class FeatureReranker:
    def __init__(self, model=None) -> None:
        self.model = model

    @property
    def trained(self) -> bool:
        return self.model is not None

    def score(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        feats = build_features(query, candidates)
        if self.model is not None:
            scores = self.model.predict_proba(feats)[:, 1]
        else:
            raw = feats @ HEURISTIC_WEIGHTS
            scores = 1.0 / (1.0 + np.exp(-raw))  # squash into (0,1) like the model
        ranked = [
            c.model_copy(update={"rerank_score": float(s)})
            for c, s in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)
        return ranked

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        if self.model is None:
            return
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> FeatureReranker:
        if not path.exists():
            return cls(None)
        try:
            import joblib

            return cls(joblib.load(path))
        except Exception as exc:  # noqa: BLE001 - a stale model must not break serving
            log.warning("reranker_load_failed", error=str(exc))
            return cls(None)


def train_reranker(features: np.ndarray, labels: np.ndarray, *, seed: int = 0):
    """Fit the stage-2 model. Pointwise on purpose: fast, calibrated, ~40 lines."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if len(np.unique(labels)) < 2:
        raise ValueError("need both positive and negative examples to train the reranker")
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.09,
        max_depth=6,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
    )
    model.fit(features, labels)
    return model


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)
