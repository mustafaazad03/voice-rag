"""Metadata-aware chunking (contextual retrieval).

Decorates any other strategy: each chunk keeps its verbatim text for display and
citation, but the *indexed* text gains a compact context header — document topic
sentence, language, answer type, position in document. A chunk that reads
"about 4 to 6 weeks" becomes findable by "how long does a passport take" because
the header carries the subject the chunk itself dropped.

Cheap version of Anthropic-style contextual retrieval: no LLM call per chunk, so
index build stays O(minutes) instead of O(dollars).
"""

from __future__ import annotations

from ..models import Chunk, Document
from .base import Chunker, ChunkingOutput, get_chunker, normalize_ws, register, split_sentences

QUERY_TYPE_HINT = {
    "DESCRIPTION": "definition",
    "NUMERIC": "number",
    "ENTITY": "entity",
    "LOCATION": "place",
    "PERSON": "person",
}


@register
class MetadataAwareChunker(Chunker):
    name = "metadata_aware"

    def __init__(self, base: str = "hierarchical", **base_params: object) -> None:
        super().__init__(base=base, **base_params)
        self._base = get_chunker(base, **base_params)
        self.needs_embedder = self._base.needs_embedder
        # Chunk ids must not collide with the undecorated strategy's ids.
        self.name = f"metadata_aware:{base}"

    def chunk(self, doc: Document) -> list[Chunk]:
        return [self._decorate(doc, c) for c in self._base.chunk(doc)]

    def chunk_documents(self, docs: list[Document]) -> ChunkingOutput:
        # Note: over `late`, the base already produced vectors from the raw text, so
        # the header improves BM25 but not the dense leg. Pair it with a base that
        # embeds downstream if you want the header in both.
        out = self._base.chunk_documents(docs)
        by_id = {d.doc_id: d for d in docs}
        chunks = [self._decorate(by_id[c.doc_id], c) for c in out.chunks]
        return ChunkingOutput(chunks=chunks, embeddings=out.embeddings)

    # ------------------------------------------------------------------ #
    def _decorate(self, doc: Document, chunk: Chunk) -> Chunk:
        header = self._header(doc, chunk)
        return chunk.model_copy(
            update={
                "strategy": self.name,
                "index_text": f"{header}\n{chunk.text}" if header else chunk.text,
                "meta": {**chunk.meta, "context_header": header},
            }
        )

    def _header(self, doc: Document, chunk: Chunk) -> str:
        bits: list[str] = []
        lead = self._lead_sentence(doc)
        if lead and lead not in chunk.text:
            bits.append(lead)
        if doc.query_type:
            hint = QUERY_TYPE_HINT.get(doc.query_type.upper())
            if hint:
                bits.append(f"type: {hint}")
        if doc.lang and doc.lang != "unknown":
            bits.append(f"lang: {doc.lang}")
        if chunk.n_chunks_in_doc > 1:
            bits.append(f"part {chunk.position + 1}/{chunk.n_chunks_in_doc}")
        return " | ".join(bits)

    @staticmethod
    def _lead_sentence(doc: Document, max_chars: int = 160) -> str:
        spans = split_sentences(doc.text)
        if not spans:
            return ""
        return normalize_ws(doc.text[spans[0][0] : spans[0][1]])[:max_chars]
