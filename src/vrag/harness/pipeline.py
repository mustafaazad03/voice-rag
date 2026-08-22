"""The harness: typed, instrumented orchestration of every pipeline stage.

Not a prompt with a retriever bolted on. Each stage has a declared input and
output type, its own span in the trace, its own failure policy, and its own place
in a latency budget that is enforced rather than hoped for:

  stage                       degrade when over budget
  ------------------------    -----------------------------------------
  guard.input                 never (safety is not optional)
  cache lookup                never (it only saves time)
  embed.query                 never (everything downstream needs it)
  retrieve.hybrid             narrower candidate lists
  rerank                      skipped, fusion order kept
  generate                    fewer sentences, extractive only
  guard.grounding             never (this is what stops hallucinations)

Failures degrade instead of exploding: a dead reranker means fusion order, a dead
LLM means the extractive generator, an ungrounded answer means the
insufficient-evidence fallback. Only a failure that would make the answer *wrong*
is allowed to surface as an error.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cache import ResponseCache, normalize_key
from ..chunking import merge_overlapping
from ..config import Settings, get_settings
from ..errors import GenerationError, QueryRejected, VRagError
from ..generate import extractive, llm
from ..guardrails import policy
from ..guardrails.grounding import check_grounding
from ..guardrails.input_guard import check_query
from ..index.embedder import Embedder, get_embedder
from ..index.store import ChunkStore
from ..models import (
    AnswerStatus,
    ConfidenceReport,
    GroundingReport,
    GuardVerdict,
    QueryRequest,
    RAGResponse,
    ScoredChunk,
    TranscriptionResult,
)
from ..obs import METRICS, Trace, get_logger
from ..retrieve.confidence import score_confidence
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import FeatureReranker
from ..stt import STTRouter

log = get_logger("harness.pipeline")


@dataclass
class Retrieved:
    """Output of the CPU-bound block, before generation."""

    candidates: list[ScoredChunk]
    confidence: ConfidenceReport
    query_vec: np.ndarray
    off_topic: bool = False
    degraded: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.degraded is None:
            self.degraded = []


class RAGPipeline:
    def __init__(
        self,
        store: ChunkStore,
        *,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: FeatureReranker | None = None,
        cache: ResponseCache | None = None,
        stt: STTRouter | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self.store = store
        self.embedder = embedder or get_embedder()
        self.retriever = HybridRetriever(store, self._s)
        self.reranker = reranker or FeatureReranker(None)
        self.cache = cache if cache is not None else ResponseCache(self._s)
        self.stt = stt if stt is not None else STTRouter(self._s)

    @classmethod
    def load(cls, settings: Settings | None = None) -> RAGPipeline:
        s = settings or get_settings()
        store = ChunkStore.load(Path(s.index_dir), s)
        reranker = FeatureReranker.load(Path(s.index_dir) / "reranker.joblib")
        log.info(
            "pipeline_ready",
            chunks=store.size,
            strategy=store.meta.get("strategy"),
            reranker="trained" if reranker.trained else "heuristic",
            stt=",".join(s.stt_providers) or "none",
        )
        return cls(store, settings=s, reranker=reranker)

    async def aclose(self) -> None:
        await self.stt.aclose()

    # ------------------------------------------------------------------ #
    # Voice entry point
    # ------------------------------------------------------------------ #
    async def answer_voice(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "",
        language: str | None = None,
        request: QueryRequest | None = None,
    ) -> RAGResponse:
        """Full voice path. STT latency is tracked separately from the query budget:
        it is a third-party network round trip, not something the pipeline controls."""
        trace = Trace()
        with trace.span("stt"):
            transcription = await self.stt.transcribe(audio, filename, content_type, language)

        base = request or QueryRequest(query=transcription.text)
        response = await self.answer(
            base.model_copy(update={"query": transcription.text}), trace=trace
        )
        # `latency_ms` stays the query-path figure; `total_ms` is what the caller
        # actually waited for, STT included. Neither number hides behind the other.
        return response.model_copy(
            update={
                "transcription": transcription,
                "stage_ms": trace.rounded(),
                "total_ms": round(trace.elapsed_ms, 3),
            }
        )

    # ------------------------------------------------------------------ #
    # Text entry point
    # ------------------------------------------------------------------ #
    async def answer(self, request: QueryRequest, trace: Trace | None = None) -> RAGResponse:
        trace = trace or Trace()
        # The query budget starts here, after STT: it covers chunk retrieval
        # through final output, which is what the latency target is about.
        budget_start = trace.elapsed_ms

        with trace.span("guard.input"):
            guard = check_query(request.query, self._s)
        if guard.verdict is GuardVerdict.BLOCK:
            METRICS.inc("query_total", status="blocked")
            return self._terminal(
                trace,
                AnswerStatus.BLOCKED,
                guard.message or "",
                guard.normalized_query,
                budget_start,
                meta={"rule": guard.rule},
            )

        query = guard.normalized_query
        cache_key = normalize_key(
            query,
            k=request.top_k or self._s.final_top_k,
            g=request.generator or self._s.generator,
            s=self.store.meta.get("strategy"),
        )

        if request.use_cache:
            with trace.span("cache.exact"):
                hit = self.cache.get(cache_key)
            if hit is not None:
                METRICS.inc("query_total", status="cache_hit")
                return hit.model_copy(
                    update={
                        "trace_id": trace.trace_id,
                        "latency_ms": round(trace.elapsed_ms - budget_start, 3),
                        "stage_ms": trace.rounded(),
                    }
                )

        # Embed first and check the semantic cache before spending anything on
        # retrieval: "what is a corporation" and "what's a corporation" miss the
        # exact cache and are the same question, which is the normal case for
        # transcribed speech.
        query_vec = await self._maybe_offload(self._embed, query, trace)

        if request.use_cache:
            with trace.span("cache.semantic"):
                hit = self.cache.get_semantic(query_vec)
            if hit is not None:
                METRICS.inc("query_total", status="cache_hit_semantic")
                return hit.model_copy(
                    update={
                        "trace_id": trace.trace_id,
                        "latency_ms": round(trace.elapsed_ms - budget_start, 3),
                        "stage_ms": trace.rounded(),
                    }
                )

        # --- CPU-bound block: retrieve -> merge -> rerank -> confidence
        retrieved = await self._maybe_offload(self._retrieve, query, query_vec, request, trace)

        if retrieved.off_topic:
            METRICS.inc("query_total", status="off_topic")
            return self._terminal(
                trace,
                AnswerStatus.OFF_TOPIC,
                policy.OFF_TOPIC,
                query,
                budget_start,
                confidence=retrieved.confidence,
                degraded=retrieved.degraded,
                meta={"reasons": retrieved.confidence.reasons},
            )

        # --- step 07: refuse before generating when the evidence is too thin
        if not retrieved.confidence.sufficient:
            METRICS.inc("query_total", status="insufficient_evidence")
            return self._terminal(
                trace,
                AnswerStatus.INSUFFICIENT_EVIDENCE,
                policy.INSUFFICIENT_EVIDENCE,
                query,
                budget_start,
                confidence=retrieved.confidence,
                degraded=retrieved.degraded,
                meta={"reasons": retrieved.confidence.reasons},
            )

        degraded = list(retrieved.degraded)
        generator = request.generator or self._s.generator
        with trace.span("generate"):
            answer, citations, gen_note = await self._generate(
                generator, query, retrieved.candidates, trace
            )
        if gen_note:
            degraded.append(gen_note)

        with trace.span("guard.grounding"):
            grounding = check_grounding(answer, retrieved.candidates, self._s)

        if not answer or not grounding.grounded:
            METRICS.inc("query_total", status="ungrounded")
            log.info(
                "ungrounded_answer_suppressed",
                trace_id=trace.trace_id,
                score=grounding.score,
                unsupported=len(grounding.unsupported_sentences),
            )
            return self._terminal(
                trace,
                AnswerStatus.INSUFFICIENT_EVIDENCE,
                policy.INSUFFICIENT_EVIDENCE,
                query,
                budget_start,
                confidence=retrieved.confidence,
                grounding=grounding,
                degraded=degraded,
            )

        latency = trace.elapsed_ms - budget_start
        response = RAGResponse(
            trace_id=trace.trace_id,
            status=AnswerStatus.ANSWERED,
            answer=answer,
            citations=citations,
            confidence=retrieved.confidence,
            grounding=grounding,
            query=query,
            latency_ms=round(latency, 3),
            stage_ms=trace.rounded(),
            degraded=degraded,
            meta={
                "strategy": self.store.meta.get("strategy"),
                "generator": generator,
                "candidates": len(retrieved.candidates),
                "redactions": guard.redactions,
            },
        )
        if request.use_cache:
            self.cache.put(cache_key, response, query_vec)
        METRICS.inc("query_total", status="answered")
        METRICS.observe("pipeline_latency", latency)
        for name, ms in trace.spans.items():
            METRICS.observe(f"stage_{name.replace('.', '_')}", ms)
        return response

    # ------------------------------------------------------------------ #
    # CPU-bound blocks
    # ------------------------------------------------------------------ #
    async def _maybe_offload(self, fn, *args):
        """Run a CPU-bound stage in a worker thread when serving, inline when not.

        ONNX Runtime and hnswlib release the GIL, so offloading keeps the event
        loop answering other requests during the ~15 ms of compute. Benchmarks set
        `offload_cpu=False` to measure the pipeline rather than the hand-off.
        """
        if self._s.offload_cpu:
            return await asyncio.to_thread(fn, *args)
        return fn(*args)

    def _embed(self, query: str, trace: Trace) -> np.ndarray:
        with trace.span("embed.query"):
            return self.embedder.encode_query(query)

    def _retrieve(
        self, query: str, query_vec: np.ndarray, request: QueryRequest, trace: Trace
    ) -> Retrieved:
        s = self._s
        degraded: list[str] = []

        # Budget check before the widest stage: shrink the candidate lists rather
        # than blowing the deadline.
        over = trace.elapsed_ms > s.budget_total_ms * 0.35
        dense_k = s.dense_top_k // 2 if over else s.dense_top_k
        sparse_k = s.sparse_top_k // 2 if over else s.sparse_top_k
        if over:
            degraded.append("narrow_candidates")

        with trace.span("retrieve.hybrid"):
            candidates = self.retriever.retrieve(
                query, query_vec, dense_k=dense_k, sparse_k=sparse_k, fusion=s.fusion
            )

        with trace.span("retrieve.merge_overlap"):
            candidates = merge_overlapping(candidates)

        rerank_mode = s.rerank
        if rerank_mode != "none" and trace.elapsed_ms > s.budget_rerank_ms:
            degraded.append("rerank_skipped_over_budget")
            METRICS.inc("degrade_total", stage="rerank")
            rerank_mode = "none"

        if rerank_mode != "none":
            with trace.span("rerank"):
                try:
                    head = self.reranker.score(query, candidates[: s.rerank_top_n])
                    candidates = head + candidates[s.rerank_top_n :]
                except Exception as exc:  # noqa: BLE001 - ranking must never 500
                    log.warning("rerank_failed", error=str(exc))
                    degraded.append("rerank_failed")

        candidates = candidates[: max(request.top_k or s.final_top_k, 1)]

        with trace.span("confidence"):
            confidence = score_confidence(query, candidates, s)

        return Retrieved(
            candidates, confidence, query_vec, off_topic=confidence.off_topic, degraded=degraded
        )

    # ------------------------------------------------------------------ #
    async def _generate(
        self, generator: str, query: str, candidates: list[ScoredChunk], trace: Trace
    ) -> tuple[str, list, str | None]:
        if generator == "llm":
            if trace.elapsed_ms > self._s.budget_generate_ms:
                # Still allowed — the caller explicitly asked for the slow path —
                # but the response says so instead of silently blowing the target.
                METRICS.inc("degrade_total", stage="generate_over_budget")
            try:
                answer, citations = await llm.generate(query, candidates, settings=self._s)
                if answer:
                    return answer, citations, "llm_over_budget"
                return "", [], "llm_refused"
            except (GenerationError, VRagError, TimeoutError) as exc:
                log.warning("llm_generate_failed", error=str(exc))
                answer, citations = extractive.generate(query, candidates, self._s)
                return answer, citations, "llm_failed_fell_back_to_extractive"

        answer, citations = extractive.generate(query, candidates, self._s)
        return answer, citations, None

    # ------------------------------------------------------------------ #
    def _terminal(
        self,
        trace: Trace,
        status: AnswerStatus,
        message: str,
        query: str,
        budget_start: float,
        *,
        confidence: ConfidenceReport | None = None,
        grounding: GroundingReport | None = None,
        degraded: list[str] | None = None,
        meta: dict | None = None,
    ) -> RAGResponse:
        latency = trace.elapsed_ms - budget_start
        METRICS.observe("pipeline_latency", latency)
        return RAGResponse(
            trace_id=trace.trace_id,
            status=status,
            answer=message,
            citations=[],
            confidence=confidence,
            grounding=grounding,
            query=query,
            latency_ms=round(latency, 3),
            stage_ms=trace.rounded(),
            degraded=degraded or [],
            meta=meta or {},
        )


__all__ = ["RAGPipeline", "QueryRejected", "TranscriptionResult"]
