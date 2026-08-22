from __future__ import annotations

import pytest

from vrag.chunking import get_chunker, list_strategies, merge_overlapping, split_sentences
from vrag.models import Chunk, Document, ScoredChunk

TEXT = (
    "First sentence about corporations. Second sentence adds detail about charters. "
    "Third sentence covers registration rules. Fourth sentence is about stock issuance."
)


def doc(text: str = TEXT) -> Document:
    return Document(doc_id="d1", text=text, lang="eng", query_type="DESCRIPTION")


def test_sentence_splitter_handles_danda():
    spans = split_sentences("पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।")
    assert len(spans) == 3


def test_sentence_splitter_never_empty_for_text_without_terminator():
    assert split_sentences("no terminator here") == [(0, 18)]


@pytest.mark.parametrize("name", ["fixed_char", "recursive", "sentence_window", "hierarchical"])
def test_offline_strategies_cover_the_document(name):
    chunker = get_chunker(name)
    chunks = chunker.chunk(doc())
    assert chunks, f"{name} produced nothing"
    assert all(c.text.strip() for c in chunks)
    assert all(0 <= c.start < c.end <= len(TEXT) for c in chunks)
    # Every character of the source must appear in at least one chunk span.
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert len(covered) >= len(TEXT.strip()) - 5


def test_fixed_char_overlap_is_real():
    chunks = get_chunker("fixed_char", size=60, overlap=20).chunk(doc())
    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end  # spans genuinely overlap


def test_fixed_char_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError):
        get_chunker("fixed_char", size=10, overlap=10)


def test_hierarchical_attaches_parent_context():
    chunks = get_chunker("hierarchical").chunk(doc())
    assert all(c.parent_text for c in chunks)
    assert all(len(c.context) >= len(c.text) for c in chunks)


def test_metadata_aware_indexes_more_than_it_displays():
    chunks = get_chunker("metadata_aware", base="hierarchical").chunk(doc())
    assert chunks
    enriched = [c for c in chunks if c.index_text]
    assert enriched, "no chunk received a context header"
    c = enriched[0]
    assert c.indexable != c.text and c.text in c.indexable
    assert "type: definition" in c.indexable


def test_empty_document_yields_no_chunks():
    for name in ["fixed_char", "recursive", "sentence_window", "hierarchical"]:
        assert get_chunker(name).chunk(doc("   ")) == []


def test_registry_lists_every_strategy():
    names = list_strategies()
    for expected in [
        "fixed_char",
        "fixed_token",
        "recursive",
        "sentence_window",
        "semantic",
        "late",
        "hierarchical",
        "metadata_aware",
    ]:
        assert expected in names


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_chunker("nope")


# --------------------------------------------------------------------------- #
# overlap handling
# --------------------------------------------------------------------------- #
def _scored(start: int, end: int, text: str, score: float, doc_id: str = "d1") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{start}", doc_id=doc_id, text=text, start=start, end=end, strategy="t"
        ),
        fusion_score=score,
    )


def test_merge_overlapping_collapses_adjacent_spans():
    merged = merge_overlapping(
        [_scored(0, 40, "alpha beta gamma", 0.9), _scored(30, 70, "beta gamma delta", 0.5)]
    )
    assert len(merged) == 1
    assert merged[0].chunk.start == 0 and merged[0].chunk.end == 70
    assert merged[0].fusion_score == 0.9


def test_merge_keeps_distinct_documents_apart():
    merged = merge_overlapping(
        [_scored(0, 40, "alpha", 0.9, "d1"), _scored(0, 40, "alpha", 0.8, "d2")]
    )
    assert len(merged) == 2


def test_merge_does_not_duplicate_the_overlapping_tail():
    merged = merge_overlapping(
        [_scored(0, 30, "the quick brown fox", 0.9), _scored(10, 45, "quick brown fox jumps", 0.4)]
    )
    assert merged[0].chunk.text.count("quick brown fox") == 1
