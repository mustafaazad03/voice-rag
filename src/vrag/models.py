"""Structured input/output contracts. Every harness stage speaks these types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    """Immutable by default: stages return new objects instead of mutating inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


class Document(Frozen):
    doc_id: str
    text: str
    lang: str = "unknown"
    source: str = "msmarco-xi"
    query_id: int | None = None
    query_type: str | None = None
    is_selected: bool = False
    trust: float = 0.5  # step 04 input: source trust prior
    meta: dict[str, Any] = Field(default_factory=dict)


class Chunk(Frozen):
    chunk_id: str
    doc_id: str
    text: str
    # Character span inside the parent document, used for overlap merging.
    start: int
    end: int
    strategy: str
    lang: str = "unknown"
    # Parent text for small-to-big retrieval; None when chunk == document.
    parent_text: str | None = None
    # What actually goes into the index when it differs from what we show the user
    # (metadata-aware strategies prepend a context header). None => use `text`.
    index_text: str | None = None
    position: int = 0
    n_chunks_in_doc: int = 1
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def context(self) -> str:
        """What the generator reads."""
        return self.parent_text or self.text

    @property
    def indexable(self) -> str:
        """What the embedder and BM25 read."""
        return self.index_text or self.text


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


class ScoredChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def score(self) -> float:
        return self.rerank_score or self.fusion_score


class Citation(Frozen):
    marker: str  # "[1]"
    chunk_id: str
    doc_id: str
    quote: str
    score: float


class ConfidenceReport(Frozen):
    """Step 04 — source confidence scoring."""

    top_score: float  # best cosine similarity in the top-k (calibrated, comparable)
    mean_top_k: float
    score_gap: float  # separation between #1 and #2, a strong ambiguity signal
    agreement: float  # fraction of top-k coming from distinct documents
    lexical_coverage: float  # share of query content terms present in evidence
    trust: float
    overall: float
    sufficient: bool
    off_topic: bool = False
    reasons: list[str] = Field(default_factory=list)


class GroundingReport(Frozen):
    """Step 07 — hallucination check."""

    grounded: bool
    score: float
    unsupported_sentences: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class GuardVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ABSTAIN = "abstain"


class GuardResult(Frozen):
    verdict: GuardVerdict
    rule: str | None = None
    message: str | None = None
    normalized_query: str = ""
    redactions: int = 0


# --------------------------------------------------------------------------- #
# Pipeline I/O
# --------------------------------------------------------------------------- #


class TranscriptionResult(Frozen):
    text: str
    provider: str
    language_code: str | None = None
    language_probability: float | None = None
    latency_ms: float = 0.0
    fallback_used: bool = False


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4096)
    top_k: int | None = None
    generator: str | None = None
    use_cache: bool = True
    language: str | None = None


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OFF_TOPIC = "off_topic"
    BLOCKED = "blocked"


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceReport | None = None
    grounding: GroundingReport | None = None
    transcription: TranscriptionResult | None = None
    query: str = ""
    # Query path only: guardrails -> embed -> retrieve -> generate -> grounding.
    latency_ms: float = 0.0
    # Voice path only: latency_ms plus the speech-to-text round trip.
    total_ms: float | None = None
    stage_ms: dict[str, float] = Field(default_factory=dict)
    cached: bool = False
    degraded: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
