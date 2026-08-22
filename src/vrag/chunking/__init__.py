"""Chunking strategies (step 01 -> index).

    fixed_char        fixed character windows with overlap (baseline)
    fixed_token       fixed subword-token windows aligned to real token boundaries
    recursive         paragraph -> sentence -> clause -> word separator hierarchy
    sentence_window   tight indexed window, wider context window returned
    semantic          embedding-distance breakpoints at topic shifts
    late              document-level context baked into per-chunk vectors
    hierarchical      small children indexed, big parent returned (default)
    metadata_aware    any of the above + contextual header in the indexed text

Pick with `--strategy`, compare with `vrag eval-chunking`.
"""

from .base import (
    Chunker,
    ChunkingOutput,
    get_chunker,
    list_strategies,
    normalize_ws,
    split_paragraphs,
    split_sentences,
)
from .overlap import merge_overlapping

__all__ = [
    "Chunker",
    "ChunkingOutput",
    "get_chunker",
    "list_strategies",
    "merge_overlapping",
    "normalize_ws",
    "split_paragraphs",
    "split_sentences",
]
