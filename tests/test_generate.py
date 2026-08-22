from __future__ import annotations

from vrag.generate import extractive
from vrag.generate.llm import REFUSAL, SYSTEM_PROMPT, _citations_from, build_prompt
from vrag.models import Chunk, ScoredChunk

CORP = (
    "A corporation is a company or group of people authorised to act as a single entity in law. "
    "Early incorporated entities were established by charter. "
    "Most jurisdictions now allow the creation of new corporations through registration."
)
PHOTO = (
    "Photosynthesis is the process used by plants to convert light energy into chemical energy. "
    "The process releases oxygen as a by-product."
)


def scored(text: str, score: float, doc_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"c-{doc_id}", doc_id=doc_id, text=text, start=0, end=len(text), strategy="t"
        ),
        dense_score=score,
        fusion_score=score,
        rerank_score=score,
    )


def test_answer_is_drawn_from_the_evidence_and_cited():
    answer, citations = extractive.generate(
        "what is a corporation", [scored(CORP, 0.9, "d1"), scored(PHOTO, 0.5, "d2")]
    )
    assert "corporation" in answer.lower()
    assert citations
    assert all(c.marker in answer for c in citations)
    # Extractive by construction: every sentence exists verbatim in some chunk.
    for part in answer.split("[")[0:1]:
        assert part.strip().rstrip("[]0123456789 ") in CORP


def test_generation_prefers_the_query_relevant_source():
    answer, citations = extractive.generate(
        "how do plants convert light", [scored(CORP, 0.6, "d1"), scored(PHOTO, 0.9, "d2")]
    )
    assert "photosynthesis" in answer.lower()
    assert citations[0].doc_id == "d2"


def test_redundant_sources_do_not_produce_a_repeated_answer():
    answer, _ = extractive.generate(
        "what is a corporation",
        [scored(CORP, 0.9, "d1"), scored(CORP, 0.88, "d2"), scored(CORP, 0.86, "d3")],
    )
    first = answer.split(".")[0]
    assert answer.count(first) == 1


def test_no_candidates_yields_no_answer():
    assert extractive.generate("anything", []) == ("", [])


def test_answer_respects_the_length_cap(settings):
    answer, _ = extractive.generate(
        "what is a corporation", [scored(CORP * 5, 0.9, "d1")], settings
    )
    assert len(answer) <= settings.answer_max_chars + 4


# --- llm generator (prompt construction only; no network) --------------------- #
def test_llm_prompt_is_context_only_and_demands_citations():
    messages = build_prompt("what is a corporation", [scored(CORP, 0.9, "d1")])
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert REFUSAL in SYSTEM_PROMPT
    assert "[1] " + CORP in messages[1]["content"]
    assert "what is a corporation" in messages[1]["content"]


def test_llm_citations_map_back_to_real_chunks():
    cands = [scored(CORP, 0.9, "d1"), scored(PHOTO, 0.8, "d2")]
    citations = _citations_from("Some claim [2] and another [1].", cands)
    assert [c.doc_id for c in citations] == ["d1", "d2"]


def test_llm_citations_ignore_markers_that_point_nowhere():
    assert _citations_from("Claim [7].", [scored(CORP, 0.9, "d1")]) == []
