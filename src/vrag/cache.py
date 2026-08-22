"""Step 09 — caching and memory.

Two layers:

  exact      normalized query string -> full response (TTL, LRU)
  semantic   query embedding -> full response when cosine >= threshold

The semantic layer is what makes voice work: two people asking the same thing get
two different transcripts ("what is a corporation" vs "what's a corporation"),
which an exact cache misses every time. Lookup is one matmul against at most a
few hundred cached vectors — tens of microseconds.
"""

from __future__ import annotations

import re
import threading
import unicodedata

import numpy as np
from cachetools import TTLCache

from .config import Settings, get_settings
from .models import RAGResponse
from .obs import METRICS

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_key(query: str, **parts: object) -> str:
    text = unicodedata.normalize("NFKC", query or "").casefold()
    text = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()
    suffix = "|".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return f"{text}|{suffix}"


class ResponseCache:
    def __init__(self, settings: Settings | None = None, semantic_capacity: int = 512) -> None:
        s = settings or get_settings()
        self._lock = threading.Lock()
        self._exact: TTLCache = TTLCache(maxsize=s.cache_size, ttl=s.cache_ttl_s)
        self._threshold = s.semantic_cache_threshold
        self._capacity = semantic_capacity
        self._vectors: np.ndarray | None = None
        self._keys: list[str] = []

    # -- exact ------------------------------------------------------------
    def get(self, key: str) -> RAGResponse | None:
        with self._lock:
            hit = self._exact.get(key)
        METRICS.inc("cache_lookup_total", layer="exact", result="hit" if hit else "miss")
        return hit.model_copy(update={"cached": True}) if hit else None

    def put(self, key: str, response: RAGResponse, vector: np.ndarray | None = None) -> None:
        with self._lock:
            self._exact[key] = response
            if vector is not None:
                self._remember(key, vector)

    # -- semantic ---------------------------------------------------------
    def get_semantic(self, vector: np.ndarray) -> RAGResponse | None:
        with self._lock:
            if self._vectors is None or not self._keys:
                METRICS.inc("cache_lookup_total", layer="semantic", result="miss")
                return None
            sims = self._vectors @ vector
            best = int(np.argmax(sims))
            if float(sims[best]) < self._threshold:
                METRICS.inc("cache_lookup_total", layer="semantic", result="miss")
                return None
            hit = self._exact.get(self._keys[best])
        METRICS.inc("cache_lookup_total", layer="semantic", result="hit" if hit else "miss")
        return hit.model_copy(update={"cached": True}) if hit else None

    def _remember(self, key: str, vector: np.ndarray) -> None:
        """Caller holds the lock."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if key in self._keys:
            return
        if self._vectors is None:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])
        self._keys.append(key)
        if len(self._keys) > self._capacity:  # FIFO eviction, mirrors the TTL layer
            drop = len(self._keys) - self._capacity
            self._keys = self._keys[drop:]
            self._vectors = self._vectors[drop:]

    def clear(self) -> None:
        with self._lock:
            self._exact.clear()
            self._vectors = None
            self._keys = []

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"exact_entries": len(self._exact), "semantic_entries": len(self._keys)}
