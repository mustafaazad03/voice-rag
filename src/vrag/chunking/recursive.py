"""Recursive splitting down a separator hierarchy.

Try to break on the most semantic boundary available (paragraph), fall back to
sentence, then clause, then hard character cut. Keeps chunks under `max_chars`
without ever slicing mid-word when a softer boundary exists.
"""

from __future__ import annotations

import re

from ..models import Chunk, Document
from .base import Chunker, register

SEPARATORS = [
    re.compile(r"\n{2,}"),  # paragraph
    re.compile(r"(?<=[.!?।॥۔؟])\s+"),  # sentence
    re.compile(r"(?<=[,;:—–])\s+"),  # clause
    re.compile(r"\s+"),  # word
]


def _recurse(text: str, offset: int, max_chars: int, depth: int) -> list[tuple[int, int]]:
    if len(text) <= max_chars or depth >= len(SEPARATORS):
        return [(offset, offset + len(text))] if text.strip() else []

    pieces: list[tuple[int, int]] = []
    cursor = 0
    for m in SEPARATORS[depth].finditer(text):
        pieces.append((cursor, m.start()))
        cursor = m.end()
    pieces.append((cursor, len(text)))
    pieces = [p for p in pieces if text[p[0] : p[1]].strip()]

    # Separator absent at this level: go deeper rather than emitting an oversized span.
    if len(pieces) <= 1:
        return _recurse(text, offset, max_chars, depth + 1)

    # Greedily glue neighbours back together up to max_chars.
    out: list[tuple[int, int]] = []
    cur_start, cur_end = pieces[0]
    for s, e in pieces[1:]:
        if e - cur_start <= max_chars:
            cur_end = e
        else:
            out.extend(_emit(text, cur_start, cur_end, offset, max_chars, depth))
            cur_start, cur_end = s, e
    out.extend(_emit(text, cur_start, cur_end, offset, max_chars, depth))
    return out


def _emit(
    text: str, start: int, end: int, offset: int, max_chars: int, depth: int
) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(offset + start, offset + end)] if text[start:end].strip() else []
    return _recurse(text[start:end], offset + start, max_chars, depth + 1)


@register
class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, max_chars: int = 640, overlap_chars: int = 96, **kw: object) -> None:
        super().__init__(max_chars=max_chars, overlap_chars=overlap_chars, **kw)
        self.max_chars, self.overlap_chars = max_chars, overlap_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        spans = _recurse(doc.text, 0, self.max_chars, 0)
        if self.overlap_chars:
            # Bleed the tail of the previous span into the next one so a fact split
            # across a boundary stays retrievable from either side.
            spans = [
                (max(0, s - self.overlap_chars) if i else s, e) for i, (s, e) in enumerate(spans)
            ]
        return [
            self.make(doc, s, e, position=i, total=len(spans))
            for i, (s, e) in enumerate(spans)
            if doc.text[s:e].strip()
        ]
