"""Semantic chunking — cut where the topic actually changes.

Embed every sentence, walk the document computing cosine distance between
consecutive sentences, and break at distances above a percentile threshold.
Unlike fixed-size splitting, a boundary lands between ideas rather than at an
arbitrary offset. Chunks are then capped so no single topic blows the encoder
window.
"""

from __future__ import annotations

import numpy as np

from ..models import Chunk, Document
from .base import Chunker, ChunkingOutput, register, split_sentences


@register
class SemanticChunker(Chunker):
    name = "semantic"
    needs_embedder = True

    def __init__(
        self,
        percentile: float = 80.0,
        min_sentences: int = 2,
        max_chars: int = 900,
        buffer: int = 1,
        **kw: object,
    ) -> None:
        super().__init__(percentile=percentile, min_sentences=min_sentences, **kw)
        self.percentile = percentile
        self.min_sentences = min_sentences
        self.max_chars = max_chars
        self.buffer = buffer  # sentences of context folded into each embedding
        self._embedder = None

    def _emb(self):
        if self._embedder is None:
            from ..index.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def chunk(self, doc: Document) -> list[Chunk]:
        return self.chunk_documents([doc]).chunks

    def chunk_documents(self, docs: list[Document]) -> ChunkingOutput:
        # Batch every sentence of every document through the encoder once.
        index: list[tuple[int, list[tuple[int, int]]]] = []
        sentences: list[str] = []
        for di, doc in enumerate(docs):
            spans = split_sentences(doc.text)
            if not spans:
                continue
            index.append((di, spans))
            for i, _span in enumerate(spans):
                lo = max(0, i - self.buffer)
                hi = min(len(spans), i + self.buffer + 1)
                sentences.append(doc.text[spans[lo][0] : spans[hi - 1][1]])

        if not sentences:
            return ChunkingOutput(chunks=[])

        vecs = self._emb().encode_passages(sentences)

        out: list[Chunk] = []
        cursor = 0
        for di, spans in index:
            doc = docs[di]
            block = vecs[cursor : cursor + len(spans)]
            cursor += len(spans)
            for pos, (s, e) in enumerate(self._boundaries(doc.text, spans, block)):
                out.append(self.make(doc, s, e, position=pos))
        # `total` is per-document; patch it in one pass.
        return ChunkingOutput(chunks=_fix_totals(out))

    def _boundaries(
        self, text: str, spans: list[tuple[int, int]], vecs: np.ndarray
    ) -> list[tuple[int, int]]:
        if len(spans) <= self.min_sentences:
            return [(spans[0][0], spans[-1][1])]

        # Vectors are already L2-normalised, so dot product is cosine.
        distances = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
        threshold = float(np.percentile(distances, self.percentile))

        groups: list[tuple[int, int]] = []
        start = 0
        for i, dist in enumerate(distances):
            too_long = spans[i][1] - spans[start][0] >= self.max_chars
            long_enough = (i - start + 1) >= self.min_sentences
            if (dist >= threshold and long_enough) or too_long:
                groups.append((spans[start][0], spans[i][1]))
                start = i + 1
        if start < len(spans):
            groups.append((spans[start][0], spans[-1][1]))
        return [g for g in groups if text[g[0] : g[1]].strip()]


def _fix_totals(chunks: list[Chunk]) -> list[Chunk]:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    return [c.model_copy(update={"n_chunks_in_doc": counts[c.doc_id]}) for c in chunks]
