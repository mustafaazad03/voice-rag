"""Hierarchical (small-to-big / parent-document) chunking.

Retrieval accuracy wants small chunks; answer quality wants large ones. Index the
small child, return the big parent. Children are recursive splits of each
paragraph; the parent is the paragraph (or the whole document when it is short).

This is the default strategy: on MS MARCO-style passages it gives the precision of
sentence-level matching without starving the generator of context.
"""

from __future__ import annotations

from ..models import Chunk, Document
from .base import Chunker, register, split_paragraphs, split_sentences


@register
class HierarchicalChunker(Chunker):
    name = "hierarchical"

    def __init__(
        self,
        child_sentences: int = 2,
        child_stride: int = 1,
        parent_max_chars: int = 1200,
        **kw: object,
    ) -> None:
        super().__init__(child_sentences=child_sentences, child_stride=child_stride, **kw)
        self.child_sentences = child_sentences
        self.child_stride = child_stride
        self.parent_max_chars = parent_max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        parents = split_paragraphs(doc.text)
        out: list[Chunk] = []
        for p_idx, (p_start, p_end) in enumerate(parents):
            parent_text = doc.text[p_start:p_end][: self.parent_max_chars]
            sents = split_sentences(doc.text[p_start:p_end])
            emitted_end = -1
            for i in range(0, len(sents), self.child_stride):
                block = sents[i : i + self.child_sentences]
                if not block:
                    continue
                c_start, c_end = p_start + block[0][0], p_start + block[-1][1]
                if c_end <= emitted_end:
                    continue
                emitted_end = c_end
                out.append(
                    self.make(
                        doc,
                        c_start,
                        c_end,
                        parent_text=parent_text,
                        position=len(out),
                        meta={"parent_index": p_idx},
                    )
                )
        return [c.model_copy(update={"n_chunks_in_doc": len(out)}) for c in out]
