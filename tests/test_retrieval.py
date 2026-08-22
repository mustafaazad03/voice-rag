from __future__ import annotations

import numpy as np

from vrag.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from vrag.index.store import ChunkStore
from vrag.ingest.normalize import Deduplicator, clean_text, fingerprint, simhash
from vrag.models import Chunk, ScoredChunk
from vrag.retrieve.confidence import score_confidence
from vrag.retrieve.hybrid import HybridRetriever
from vrag.retrieve.rerank import FEATURE_NAMES, FeatureReranker, build_features


def scored(text: str, score: float, doc_id: str = "d1", trust: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c{abs(hash(text)) % 9999}",
            doc_id=doc_id,
            text=text,
            start=0,
            end=len(text),
            strategy="t",
            meta={"trust": trust},
        ),
        dense_score=score,
        fusion_score=score,
        dense_rank=0,
    )


# --- hybrid retrieval -------------------------------------------------------- #
def test_hybrid_finds_the_lexically_exact_passage(store, embedder, settings):
    r = HybridRetriever(store, settings)
    query = "boiling point of water at sea level"
    hits = r.retrieve(query, embedder.encode_query(query))
    assert hits
    assert "boiling" in hits[0].chunk.text.casefold()


def test_hybrid_finds_a_paraphrase_with_no_shared_rare_terms(store, embedder, settings):
    r = HybridRetriever(store, settings)
    query = "how do plants turn sunlight into energy"
    hits = r.retrieve(query, embedder.encode_query(query))
    top = " ".join(h.chunk.text.casefold() for h in hits[:3])
    assert "photosynthesis" in top


def test_both_legs_contribute(store, embedder, settings):
    r = HybridRetriever(store, settings)
    query = "corporation registration"
    hits = r.retrieve(query, embedder.encode_query(query))
    assert any(h.dense_rank is not None for h in hits)
    assert any(h.sparse_rank is not None for h in hits)


def test_weighted_fusion_also_works(store, embedder, settings):
    r = HybridRetriever(store, settings)
    query = "what is a corporation"
    assert r.retrieve(query, embedder.encode_query(query), fusion="weighted")


# --- reranker ----------------------------------------------------------------- #
def test_feature_matrix_shape_matches_the_declared_features():
    feats = build_features("what is a corporation", [scored("a corporation is a company", 0.8)])
    assert feats.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(feats).all()


def test_heuristic_reranker_prefers_the_lexically_covering_chunk():
    ranked = FeatureReranker(None).score(
        "boiling point of water",
        [
            scored("chlorophyll absorbs blue and red light", 0.55),
            scored("the boiling point of water at sea level is 100 degrees", 0.54),
        ],
    )
    assert "boiling" in ranked[0].chunk.text


def test_reranker_on_empty_candidates_is_empty():
    assert FeatureReranker(None).score("q", []) == []


# --- confidence ---------------------------------------------------------------- #
def test_confidence_is_high_for_strong_multi_source_evidence(settings):
    report = score_confidence(
        "what is a corporation",
        [
            scored(
                "a corporation is a company authorised to act as a single entity", 0.92, "d1", 0.85
            ),
            scored("a corporation may issue stock and is governed by law", 0.71, "d2", 0.85),
        ],
        settings,
    )
    assert report.sufficient
    assert report.agreement == 1.0
    assert report.lexical_coverage > 0.3


def test_confidence_is_insufficient_without_candidates(settings):
    report = score_confidence("anything", [], settings)
    assert not report.sufficient
    assert report.reasons == ["no_candidates"]


def test_confidence_calls_a_distant_query_off_topic(settings):
    report = score_confidence(
        "what is quantum chromodynamics", [scored("unrelated text about trees", 0.05)], settings
    )
    assert not report.sufficient
    assert report.off_topic
    assert "off_topic" in report.reasons


def test_confidence_separates_on_topic_but_weak_from_off_topic(settings):
    # Similarity sits between off_topic_similarity (0.80) and min_retrieval_score (0.84):
    # the corpus covers the subject, the evidence is still too thin to answer.
    report = score_confidence(
        "what is a corporation",
        [scored("a corporation filing calendar and unrelated procedural notes", 0.82)],
        settings,
    )
    assert not report.sufficient
    assert not report.off_topic
    assert "weak_top_score" in report.reasons


# --- ingest normalisation --------------------------------------------------------- #
def test_clean_text_collapses_whitespace_and_control_characters():
    assert clean_text("a\x00b   c\n\nd") == "a b c d"


def test_fingerprint_ignores_case_and_spacing():
    assert fingerprint("Hello   World") == fingerprint("hello world")


def test_deduplicator_drops_exact_and_near_duplicates():
    dd = Deduplicator(max_distance=14)
    base = (
        "A corporation is a company or group of people authorised to act as a single entity in law."
    )
    assert dd.is_duplicate(base) is False
    assert dd.is_duplicate(base.upper()) is True  # exact after normalisation
    assert dd.is_duplicate(base + " Extra trailing note.") is True  # near duplicate
    assert (
        dd.is_duplicate("Photosynthesis converts light energy into chemical energy in plants.")
        is False
    )


def test_simhash_of_similar_texts_is_close():
    a = simhash("the quick brown fox jumps over the lazy dog")
    b = simhash("the quick brown fox jumps over the lazy dog today")
    assert bin(a ^ b).count("1") <= 12


# --- metrics ---------------------------------------------------------------------- #
def test_ranking_metrics_are_sane():
    retrieved, relevant = ["a", "b", "c"], {"b"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert mrr_at_k(retrieved, relevant, 3) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) < 1
    assert ndcg_at_k(["b"], {"b"}, 3) == 1.0


# --- index persistence -------------------------------------------------------- #
def test_store_survives_a_save_load_roundtrip(store, embedder, settings, tmp_path):
    path = tmp_path / "index"
    store.save(path)
    reloaded = ChunkStore.load(path, settings)

    assert reloaded.size == store.size
    assert reloaded.meta["strategy"] == store.meta["strategy"]

    query = "what is a corporation"
    vec = embedder.encode_query(query)
    before = [c.chunk.chunk_id for c in HybridRetriever(store, settings).retrieve(query, vec)]
    after = [c.chunk.chunk_id for c in HybridRetriever(reloaded, settings).retrieve(query, vec)]
    assert before == after
