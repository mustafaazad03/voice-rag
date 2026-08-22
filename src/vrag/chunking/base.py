"""Chunker contract, registry and shared text-splitting utilities.

Splitting is script-aware: the corpus is Indic, so the sentence terminator set
includes danda/double-danda and the Urdu full stop alongside ASCII punctuation.
"""

from __future__ import annotations

import abc
import hashlib
import re
from dataclasses import dataclass

import numpy as np

from ..models import Chunk, Document

# Danda, double danda, Urdu full stop, ASCII terminators, ideographic stop.
SENTENCE_END = r"[.!?।॥۔؟…]"
# Single-char lookbehind (fixed width); repeated terminators stay with the sentence.
_SENT_RE = re.compile(rf"(?<={SENTENCE_END})[\"'’”)\]]?\s+|\n{{2,}}")
_PARA_RE = re.compile(r"\n{2,}|\r\n{2,}")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence char spans. Never returns empty for non-empty input."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _SENT_RE.finditer(text):
        end = m.start()
        if end > cursor and text[cursor:end].strip():
            spans.append((cursor, end))
        cursor = m.end()
    if cursor < len(text) and text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans or ([(0, len(text))] if text.strip() else [])


def split_paragraphs(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _PARA_RE.finditer(text):
        if text[cursor : m.start()].strip():
            spans.append((cursor, m.start()))
        cursor = m.end()
    if text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans or ([(0, len(text))] if text.strip() else [])


def chunk_id_for(doc_id: str, start: int, end: int, strategy: str) -> str:
    digest = hashlib.blake2b(
        f"{doc_id}:{start}:{end}:{strategy}".encode(), digest_size=8
    ).hexdigest()
    return f"{strategy[:4]}_{digest}"


@dataclass(frozen=True)
class ChunkingOutput:
    chunks: list[Chunk]
    # Only late chunking fills this in; everything else embeds normally downstream.
    embeddings: np.ndarray | None = None


class Chunker(abc.ABC):
    """One strategy for turning documents into retrievable units."""

    name: str = "base"
    #: Strategies that need the encoder at index-build time.
    needs_embedder: bool = False

    def __init__(self, **params: object) -> None:
        self.params = params

    @abc.abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    def chunk_documents(self, docs: list[Document]) -> ChunkingOutput:
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk(doc))
        return ChunkingOutput(chunks=out)

    # -- helpers used by subclasses ---------------------------------------
    def make(
        self,
        doc: Document,
        start: int,
        end: int,
        *,
        parent_text: str | None = None,
        position: int = 0,
        total: int = 1,
        text: str | None = None,
        meta: dict | None = None,
    ) -> Chunk:
        body = text if text is not None else doc.text[start:end]
        return Chunk(
            chunk_id=chunk_id_for(doc.doc_id, start, end, self.name),
            doc_id=doc.doc_id,
            text=normalize_ws(body),
            start=start,
            end=end,
            strategy=self.name,
            lang=doc.lang,
            parent_text=parent_text,
            position=position,
            n_chunks_in_doc=total,
            # Carry the document's trust prior onto the chunk: confidence scoring
            # (step 04) runs on chunks and never sees the Document again.
            meta={"trust": doc.trust, **(meta or {})},
        )


REGISTRY: dict[str, type[Chunker]] = {}


def register(cls: type[Chunker]) -> type[Chunker]:
    REGISTRY[cls.name] = cls
    return cls


def get_chunker(name: str, **params: object) -> Chunker:
    from . import (  # noqa: F401
        fixed,
        hierarchical,
        late,
        metadata_aware,
        recursive,
        semantic,
        sentence_window,
    )

    if name not in REGISTRY:
        raise KeyError(f"unknown chunk strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def list_strategies() -> list[str]:
    from . import (  # noqa: F401
        fixed,
        hierarchical,
        late,
        metadata_aware,
        recursive,
        semantic,
        sentence_window,
    )

    return sorted(REGISTRY)
