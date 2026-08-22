"""BM25 lexical index (step 02, sparse half).

bm25s keeps the whole thing in scipy sparse matrices, so a top-60 lookup over a
few hundred thousand chunks costs ~1-3 ms — cheap enough to always run alongside
the dense leg instead of choosing between them.

Tokenization is deliberately stopword-free and unstemmed: the corpus spans 14
Indic scripts plus English, and an English stemmer would quietly damage the rest.
"""

from __future__ import annotations

from pathlib import Path

import bm25s
import numpy as np

from ..obs import get_logger

log = get_logger("index.sparse")

# Unicode-aware: keeps Devanagari/Bengali/Tamil words, drops punctuation.
SPLIT_PATTERN = r"(?u)\w+"


class SparseIndex:
    def __init__(self, retriever: bm25s.BM25 | None = None) -> None:
        self._bm25 = retriever
        self.size = 0

    @classmethod
    def build(cls, texts: list[str]) -> SparseIndex:
        tokens = bm25s.tokenize(
            texts, stopwords=None, stemmer=None, token_pattern=SPLIT_PATTERN, show_progress=False
        )
        retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        retriever.index(tokens, show_progress=False)
        idx = cls(retriever)
        idx.size = len(texts)
        log.info("sparse_index_built", chunks=len(texts))
        return idx

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (chunk_indices, scores), best first."""
        if self._bm25 is None or self.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        tokens = bm25s.tokenize(
            query, stopwords=None, stemmer=None, token_pattern=SPLIT_PATTERN, show_progress=False
        )
        k = min(k, self.size)
        try:
            ids, scores = self._bm25.retrieve(tokens, k=k, show_progress=False, n_threads=1)
        except ValueError:
            # Query has no token present in the vocabulary.
            return np.empty(0, np.int64), np.empty(0, np.float32)
        return ids[0].astype(np.int64), scores[0].astype(np.float32)

    def save(self, path: Path) -> None:
        assert self._bm25 is not None
        self._bm25.save(str(path), allow_pickle=True)

    @classmethod
    def load(cls, path: Path, size: int) -> SparseIndex:
        idx = cls(bm25s.BM25.load(str(path), mmap=False, allow_pickle=True))
        idx.size = size
        return idx
