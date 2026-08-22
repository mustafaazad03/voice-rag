"""Late chunking (Gunther et al., 2024 — "Late Chunking of Short Chunks").

Ordinary chunking embeds each chunk in isolation, so a chunk saying "it was
founded in 1998" loses whatever "it" referred to. Late chunking runs the encoder
over the *whole* document first, then mean-pools the contextualised token vectors
per chunk span. Same chunk text, but the vector remembers the document.

Costs one extra forward pass at index time and nothing at query time.
"""

from __future__ import annotations

import numpy as np

from ..models import Chunk, Document
from .base import Chunker, ChunkingOutput, register, split_sentences


@register
class LateChunker(Chunker):
    name = "late"
    needs_embedder = True

    def __init__(
        self, window: int = 2, stride: int = 2, max_tokens: int = 512, **kw: object
    ) -> None:
        super().__init__(window=window, stride=stride, **kw)
        self.window, self.stride, self.max_tokens = window, stride, max_tokens
        self._embedder = None
        self._tok = None

    def _emb(self):
        if self._embedder is None:
            from ..index.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _tokenizer(self):
        if self._tok is None:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            from ..config import get_settings

            s = get_settings()
            tok = Tokenizer.from_file(hf_hub_download(s.embed_repo, s.embed_tokenizer_file))
            tok.enable_truncation(self.max_tokens)
            self._tok = tok
        return self._tok

    def chunk(self, doc: Document) -> list[Chunk]:
        return self.chunk_documents([doc]).chunks

    def chunk_documents(self, docs: list[Document], batch_size: int = 16) -> ChunkingOutput:
        from ..config import get_settings

        prefix = get_settings().embed_passage_prefix
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []

        for start in range(0, len(docs), batch_size):
            batch = [d for d in docs[start : start + batch_size] if d.text.strip()]
            if not batch:
                continue
            c, v = self._process(batch, prefix)
            chunks.extend(c)
            if len(v):
                vectors.append(v)

        embeddings = np.vstack(vectors) if vectors else np.zeros((0, self._emb().dim), np.float32)
        return ChunkingOutput(chunks=chunks, embeddings=embeddings)

    # ------------------------------------------------------------------ #
    def _process(self, docs: list[Document], prefix: str) -> tuple[list[Chunk], np.ndarray]:
        encs = self._tokenizer().encode_batch([prefix + d.text for d in docs])
        width = max(len(e.ids) for e in encs)
        ids = np.ones((len(encs), width), dtype=np.int64)  # 1 == <pad> for XLM-R
        mask = np.zeros((len(encs), width), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.attention_mask)] = e.attention_mask

        hidden = self._emb().forward_ids(ids, mask)  # (B, T, D) — full-document context

        out_chunks: list[Chunk] = []
        out_vecs: list[np.ndarray] = []
        shift = len(prefix)

        for i, (doc, enc) in enumerate(zip(docs, encs, strict=True)):
            # Token index -> char span in the original document text.
            offsets = [
                (idx, (s - shift, e - shift))
                for idx, (s, e) in enumerate(enc.offsets)
                if e > s and s >= shift
            ]
            if not offsets:
                continue
            spans = self._spans(doc, offsets)
            for pos, (cstart, cend, tok_lo, tok_hi) in enumerate(spans):
                vec = hidden[i, tok_lo : tok_hi + 1].mean(axis=0)
                out_vecs.append(vec)
                out_chunks.append(self.make(doc, cstart, cend, position=pos, total=len(spans)))

        if not out_vecs:
            return [], np.zeros((0, self._emb().dim), np.float32)
        vecs = np.vstack(out_vecs).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return out_chunks, vecs

    def _spans(
        self, doc: Document, offsets: list[tuple[int, tuple[int, int]]]
    ) -> list[tuple[int, int, int, int]]:
        """Sentence-window spans expressed as (char_start, char_end, tok_start, tok_end)."""
        sents = split_sentences(doc.text)
        if not sents:
            return []
        covered = offsets[-1][1][1]  # truncation may cut the document short
        out: list[tuple[int, int, int, int]] = []
        for i in range(0, len(sents), self.stride):
            block = sents[i : i + self.window]
            if not block:
                continue
            cstart, cend = block[0][0], min(block[-1][1], covered)
            if cend <= cstart:
                break
            inside = [idx for idx, (s, e) in offsets if s >= cstart and e <= cend]
            if not inside:
                continue
            if out and cend <= out[-1][1]:
                continue
            out.append((cstart, cend, inside[0], inside[-1]))
        return out
