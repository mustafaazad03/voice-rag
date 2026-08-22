"""Step 01 — normalize and deduplicate.

MS MARCO passages repeat heavily across queries: the same paragraph is attached
to dozens of questions. Left alone that inflates the index, skews BM25 IDF and
fills top-k with copies of one source. Two passes:

  exact    blake2b of the normalized text          (catches whitespace/case noise)
  near     64-bit SimHash + banded lookup          (catches boilerplate variants)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from ..models import Document

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def clean_text(text: str) -> str:
    """NFKC, strip control chars and collapse whitespace. Content is preserved."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CTRL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def fingerprint(text: str) -> str:
    canon = _WS_RE.sub(" ", _URL_RE.sub("", text.casefold())).strip()
    return hashlib.blake2b(canon.encode(), digest_size=16).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    """Charikar SimHash over word shingles. Near-identical text -> near-identical hash."""
    tokens = _TOKEN_RE.findall(text.casefold())
    if not tokens:
        return 0
    shingles = (
        [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)] if len(tokens) > 2 else tokens
    )
    vector = [0] * bits
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, v in enumerate(vector):
        if v > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Deduplicator:
    """Streaming dedup. `max_distance=0` disables the near-duplicate pass.

    Calibration note: with 3-gram shingles, editing one word of a short passage
    moves roughly 12 of 64 bits, while unrelated passages sit near 29. 12 is the
    usable separation point; 16 narrow bands keep LSH recall high at that radius
    (a wider band is cheaper but starts missing true duplicates).
    """

    max_distance: int = 12
    bands: int = 16
    _seen: set[str] = field(default_factory=set, init=False)
    _buckets: dict[tuple[int, int], list[int]] = field(default_factory=dict, init=False)
    exact_dropped: int = field(default=0, init=False)
    near_dropped: int = field(default=0, init=False)

    def is_duplicate(self, text: str) -> bool:
        fp = fingerprint(text)
        if fp in self._seen:
            self.exact_dropped += 1
            return True
        self._seen.add(fp)

        if self.max_distance <= 0:
            return False

        # Banded lookup: identical hashes in any band are the only viable candidates.
        h = simhash(text)
        width = 64 // self.bands
        keys = [(b, (h >> (b * width)) & ((1 << width) - 1)) for b in range(self.bands)]
        for key in keys:
            for other in self._buckets.get(key, ()):
                if hamming(h, other) <= self.max_distance:
                    self.near_dropped += 1
                    return True
        for key in keys:
            self._buckets.setdefault(key, []).append(h)
        return False

    @property
    def stats(self) -> dict[str, int]:
        return {
            "unique": len(self._seen) - self.near_dropped,
            "exact_dropped": self.exact_dropped,
            "near_dropped": self.near_dropped,
        }


def dedupe_documents(docs: list[Document], max_distance: int = 12) -> tuple[list[Document], dict]:
    dd = Deduplicator(max_distance=max_distance)
    kept = [d for d in docs if not dd.is_duplicate(d.text)]
    return kept, dd.stats
