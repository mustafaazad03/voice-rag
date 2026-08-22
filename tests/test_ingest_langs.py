"""Language shard selection. No network: the dataset call is the only part that
needs it, and these cover everything around it."""

from __future__ import annotations

import pytest

from vrag.ingest.loader import LANGS, _interleave, _shard_uri, lang_code


def test_shard_uri_points_at_the_per_language_parquet():
    assert _shard_uri("validation", "hin").endswith("/validation/hinval.parquet")


def test_every_declared_language_has_a_distinct_shard():
    uris = {_shard_uri("validation", lang) for lang in LANGS}
    assert len(uris) == len(LANGS) == 14


def test_interleave_spreads_a_query_budget_across_languages():
    """A concatenated stream would spend the whole budget on the first language."""
    streams = [iter(["hin1", "hin2", "hin3"]), iter(["tam1", "tam2"]), iter(["ben1"])]
    assert list(_interleave(streams)) == ["hin1", "tam1", "ben1", "hin2", "tam2", "hin3"]


def test_interleave_survives_streams_of_unequal_length():
    assert list(_interleave([iter([1]), iter([2, 3, 4])])) == [1, 2, 3, 4]


def test_unknown_language_is_rejected_before_any_download():
    from vrag.ingest.loader import _stream

    with pytest.raises(ValueError, match="unknown language"):
        next(iter(_stream("validation", ["klingon"])))


def test_lang_code_normalises_the_dataset_tags():
    assert lang_code("hin_Deva") == "hin"
    assert lang_code("asm_Beng") == "asm"
