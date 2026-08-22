"""HNSW approximate nearest neighbour index (step 03, ANN half).

Vectors are L2-normalised upstream, so inner product == cosine and we can use the
cheaper `ip` space. `ef_search` is the accuracy/latency dial: 64 keeps recall@60
near-exact on this corpus at well under a millisecond.
"""

from __future__ import annotations

from pathlib import Path

import hnswlib
import numpy as np

from ..obs import get_logger

log = get_logger("index.dense")


class DenseIndex:
    def __init__(self, index: hnswlib.Index | None, dim: int, size: int = 0) -> None:
        self._index = index
        self.dim = dim
        self.size = size

    @classmethod
    def build(
        cls, vectors: np.ndarray, *, m: int = 32, ef_construction: int = 200, ef_search: int = 64
    ) -> DenseIndex:
        n, dim = vectors.shape
        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        index.add_items(np.ascontiguousarray(vectors, dtype=np.float32), np.arange(n))
        index.set_ef(max(ef_search, 16))
        log.info("dense_index_built", vectors=n, dim=dim, M=m)
        return cls(index, dim, n)

    def set_ef(self, ef: int) -> None:
        if self._index is not None:
            self._index.set_ef(max(ef, 8))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (chunk_indices, cosine similarities), best first."""
        if self._index is None or self.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        k = min(k, self.size)
        labels, distances = self._index.knn_query(query.reshape(1, -1).astype(np.float32), k=k)
        # hnswlib `ip` distance is 1 - inner_product.
        return labels[0].astype(np.int64), (1.0 - distances[0]).astype(np.float32)

    def save(self, path: Path) -> None:
        assert self._index is not None
        self._index.save_index(str(path))

    @classmethod
    def load(cls, path: Path, dim: int, size: int, ef_search: int = 64) -> DenseIndex:
        index = hnswlib.Index(space="ip", dim=dim)
        index.load_index(str(path), max_elements=size)
        index.set_ef(max(ef_search, 16))
        return cls(index, dim, size)
