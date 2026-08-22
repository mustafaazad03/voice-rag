"""Sentence-window chunking.

Index a tight window of sentences (precise embedding, high lexical density) but
carry a wider neighbourhood as `parent_text` so generation still sees the context
around the match. Stride < window gives sentence-level overlap.
"""

from __future__ import annotations

from ..models import Chunk, Document
from .base import Chunker, register, split_sentences


@register
class SentenceWindowChunker(Chunker):
    name = "sentence_window"

    def __init__(self, window: int = 3, stride: int = 2, context: int = 2, **kw: object) -> None:
        super().__init__(window=window, stride=stride, context=context, **kw)
        if stride < 1 or window < 1:
            raise ValueError("window and stride must be >= 1")
        self.window, self.stride, self.context = window, stride, context

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        if not sents:
            return []

        starts = list(range(0, len(sents), self.stride))
        # Drop windows entirely contained in the previous one (short tail docs).
        windows = []
        for i in starts:
            block = sents[i : i + self.window]
            if not block:
                continue
            span = (block[0][0], block[-1][1])
            if windows and span[1] <= windows[-1][1][1]:
                continue
            windows.append((i, span))

        out: list[Chunk] = []
        for pos, (i, (s, e)) in enumerate(windows):
            lo = max(0, i - self.context)
            hi = min(len(sents), i + self.window + self.context)
            parent = doc.text[sents[lo][0] : sents[hi - 1][1]]
            out.append(
                self.make(
                    doc,
                    s,
                    e,
                    parent_text=parent,
                    position=pos,
                    total=len(windows),
                    meta={"n_sentences": min(self.window, len(sents) - i)},
                )
            )
        return out
