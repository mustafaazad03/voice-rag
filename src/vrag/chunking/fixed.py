"""Fixed-size chunking — the baseline every other strategy is measured against.

Two variants: real subword tokens (accurate, matches the encoder window) and
characters (fast, no tokenizer round-trip). Both support overlap.
"""

from __future__ import annotations

from ..models import Chunk, Document
from .base import Chunker, register


@register
class FixedCharChunker(Chunker):
    name = "fixed_char"

    def __init__(self, size: int = 700, overlap: int = 120, **kw: object) -> None:
        super().__init__(size=size, overlap=overlap, **kw)
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap = size, overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        stride = self.size - self.overlap
        bounds = [(i, min(i + self.size, len(text))) for i in range(0, max(len(text), 1), stride)]
        # Drop a trailing window fully covered by its predecessor.
        bounds = [b for i, b in enumerate(bounds) if i == 0 or b[1] > bounds[i - 1][1]]
        return [
            self.make(doc, s, e, position=i, total=len(bounds))
            for i, (s, e) in enumerate(bounds)
            if text[s:e].strip()
        ]


@register
class FixedTokenChunker(Chunker):
    """Fixed subword-token windows, aligned to real token boundaries."""

    name = "fixed_token"

    def __init__(self, size: int = 160, overlap: int = 32, **kw: object) -> None:
        super().__init__(size=size, overlap=overlap, **kw)
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap = size, overlap
        self._tok = None

    def _tokenizer(self):
        # A separate, un-truncated tokenizer: here we are splitting, not encoding.
        if self._tok is None:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            from ..config import get_settings

            s = get_settings()
            self._tok = Tokenizer.from_file(hf_hub_download(s.embed_repo, s.embed_tokenizer_file))
        return self._tok

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        enc = self._tokenizer().encode(doc.text)
        offsets = [o for o in enc.offsets if o != (0, 0)]
        if not offsets:
            return []
        stride = self.size - self.overlap
        windows = [offsets[i : i + self.size] for i in range(0, len(offsets), stride)]
        windows = [w for w in windows if w]
        out: list[Chunk] = []
        for i, win in enumerate(windows):
            start, end = win[0][0], win[-1][1]
            if doc.text[start:end].strip():
                out.append(self.make(doc, start, end, position=i, total=len(windows)))
        return out
