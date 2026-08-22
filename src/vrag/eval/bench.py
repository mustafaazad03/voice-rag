"""Latency benchmark — the P50/P70/P100 numbers the brief asks for.

Measured over the *query path*: input guardrails, query embedding, hybrid
retrieval, overlap merge, rerank, confidence, generation, grounding check. That
is "chunking + vector DB retrieval + everything through to final output" —
chunking itself is an index-build cost, reported separately by `vrag index-info`.

Speech-to-text is reported as its own number and excluded from the pipeline
figure, because it is a third-party network round trip: including it would
measure Sarvam's datacentre distance, not this pipeline.

Cache is disabled by default so every run is a cold path. `--warm` measures the
cached path for comparison.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import get_settings
from ..harness.pipeline import RAGPipeline
from ..models import AnswerStatus, QueryRequest
from ..obs import percentiles

PERCENTILES = (50, 70, 90, 95, 99, 100)


@dataclass
class BenchResult:
    queries: int
    warmup: int
    cache: bool
    latency_ms: dict[str, float]
    stage_ms: dict[str, dict[str, float]]
    mean_ms: float
    stdev_ms: float
    under_target: float
    target_ms: float
    status_counts: dict[str, int]
    index: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lat = self.latency_ms
        head = "| P50 | P70 | P90 | P95 | P99 | P100 |\n|---|---|---|---|---|---|\n"
        row = "| " + " | ".join(f"{lat[str(p)]:.1f} ms" for p in PERCENTILES) + " |\n"
        lines = [
            f"**{self.queries} queries** · cache {'on' if self.cache else 'off'} · "
            f"target {self.target_ms:.0f} ms\n",
            head + row,
            f"\nmean {self.mean_ms:.1f} ms · stdev {self.stdev_ms:.1f} ms · "
            f"{self.under_target * 100:.1f}% under target\n",
            "\n| stage | P50 | P95 | P100 |\n|---|---|---|---|",
        ]
        for stage, p in sorted(self.stage_ms.items(), key=lambda kv: -kv[1]["50"]):
            lines.append(f"| {stage} | {p['50']:.2f} | {p['95']:.2f} | {p['100']:.2f} |")
        lines.append(f"\nstatus: {json.dumps(self.status_counts)}")
        return "\n".join(lines)


async def run_benchmark(
    *,
    n: int = 200,
    warmup: int = 20,
    use_cache: bool = False,
    seed: int = 7,
    queries: list[str] | None = None,
    pipeline: RAGPipeline | None = None,
) -> BenchResult:
    s = get_settings()
    # Benchmarks measure the pipeline, not the thread hand-off.
    s.offload_cpu = False
    pipe = pipeline or RAGPipeline.load(s)

    pool = queries or _load_queries(Path(s.index_dir).parent / "corpus", seed=seed)
    if not pool:
        raise RuntimeError("no benchmark queries available; run `vrag ingest` first")

    rng = random.Random(seed)
    sample = [rng.choice(pool) for _ in range(n + warmup)]

    latencies: list[float] = []
    stages: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}

    for i, q in enumerate(sample):
        if not use_cache:
            pipe.cache.clear()
        resp = await pipe.answer(QueryRequest(query=q, use_cache=use_cache))
        if i < warmup:
            continue
        latencies.append(resp.latency_ms)
        for name, ms in resp.stage_ms.items():
            stages.setdefault(name, []).append(ms)
        key = resp.status.value if isinstance(resp.status, AnswerStatus) else str(resp.status)
        statuses[key] = statuses.get(key, 0) + 1

    target = s.budget_total_ms
    return BenchResult(
        queries=len(latencies),
        warmup=warmup,
        cache=use_cache,
        latency_ms=percentiles(latencies, PERCENTILES),
        stage_ms={k: percentiles(v, PERCENTILES) for k, v in stages.items()},
        mean_ms=round(statistics.fmean(latencies), 3),
        stdev_ms=round(statistics.pstdev(latencies), 3) if len(latencies) > 1 else 0.0,
        under_target=round(sum(1 for x in latencies if x <= target) / len(latencies), 4),
        target_ms=target,
        status_counts=statuses,
        index=pipe.store.meta,
    )


def _load_queries(corpus_dir: Path, *, seed: int = 7, limit: int = 3000) -> list[str]:
    if not (corpus_dir / "queries.jsonl").exists():
        return []
    corpus_queries = [
        json.loads(line)["query"]
        for line in (corpus_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rng = random.Random(seed)
    rng.shuffle(corpus_queries)
    return corpus_queries[:limit]


def main(
    n: int = 200, warmup: int = 20, use_cache: bool = False, out: Path | None = None
) -> BenchResult:
    result = asyncio.run(run_benchmark(n=n, warmup=warmup, use_cache=use_cache))
    if out:
        out.write_text(json.dumps(asdict(result), indent=2))
    return result
