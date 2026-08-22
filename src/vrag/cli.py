"""vrag command line."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import typer

from .config import REPO_ROOT, get_settings
from .obs import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="Voice-enabled RAG pipeline.")
log = get_logger("cli")

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
REPORT_DIR = REPO_ROOT / "reports"


@app.callback()
def _root() -> None:
    configure_logging()


# --------------------------------------------------------------------------- #
@app.command()
def ingest(
    queries: int = typer.Option(3000, help="How many source queries to stream from the dataset."),
    langs: str = typer.Option("", help="Comma-separated language filter, e.g. hin,ben,tam."),
    split: str = typer.Option("validation"),
    text_field: str = typer.Option("translated", help="translated | english | both"),
    strategy: str = typer.Option("", help="Chunking strategy (default: config)."),
    train_rerank: bool = typer.Option(True, help="Fit the stage-2 reranker on is_selected labels."),
    dedupe_distance: int = typer.Option(12, help="SimHash Hamming radius; 0 disables near-dedup."),
) -> None:
    """Stream MSMARCO-XI, normalize, chunk, index, and train the reranker."""
    from .index.embedder import get_embedder
    from .index.store import ChunkStore
    from .ingest.loader import load_corpus

    s = get_settings()
    corpus = load_corpus(
        limit_queries=queries,
        langs=[x for x in langs.split(",") if x] or None,
        split=split,
        text_field=text_field,
        dedupe_distance=dedupe_distance,
    )
    corpus.save(CORPUS_DIR)
    typer.echo(json.dumps(corpus.stats, indent=2, ensure_ascii=False))

    embedder = get_embedder()
    store = ChunkStore.build(
        corpus.documents, strategy=strategy or s.chunk_strategy, settings=s, embedder=embedder
    )
    store.save(Path(s.index_dir))

    if train_rerank:
        from .retrieve.train import fit

        try:
            reranker = fit(store, embedder, corpus.queries, s)
            reranker.save(Path(s.index_dir) / "reranker.joblib")
            typer.echo("reranker: trained")
        except ValueError as exc:
            typer.echo(f"reranker: skipped ({exc}) — serving falls back to heuristic weights")

    typer.echo(json.dumps(store.meta, indent=2))


# --------------------------------------------------------------------------- #
@app.command("train-rerank")
def train_rerank(
    max_queries: int = typer.Option(1500, help="Labelled queries to train on."),
) -> None:
    """Fit the stage-2 reranker against the existing index, without re-ingesting."""
    from .index.embedder import get_embedder
    from .index.store import ChunkStore
    from .ingest.loader import Corpus
    from .retrieve.train import fit

    s = get_settings()
    store = ChunkStore.load(Path(s.index_dir), s)
    corpus = Corpus.load(CORPUS_DIR)
    reranker = fit(store, get_embedder(), corpus.queries, s, max_queries=max_queries)
    reranker.save(Path(s.index_dir) / "reranker.joblib")
    typer.echo(f"saved {Path(s.index_dir) / 'reranker.joblib'}")


# --------------------------------------------------------------------------- #
@app.command()
def query(
    text: str,
    top_k: int = typer.Option(0),
    generator: str = typer.Option("", help="extractive | llm"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Answer one question from the text entry point."""
    from .harness.pipeline import RAGPipeline
    from .models import QueryRequest

    pipe = RAGPipeline.load()
    req = QueryRequest(
        query=text,
        top_k=top_k or None,
        generator=generator or None,
        use_cache=False,
    )
    resp = asyncio.run(pipe.answer(req))
    if json_out:
        typer.echo(resp.model_dump_json(indent=2))
        return
    typer.echo(f"\n{resp.status.value.upper()}  ({resp.latency_ms:.1f} ms)\n")
    typer.echo(resp.answer)
    for c in resp.citations:
        typer.echo(f"  {c.marker} {c.doc_id} · {c.score:.3f} · {c.quote[:120]}")
    if resp.confidence:
        typer.echo(f"\nconfidence {resp.confidence.overall:.3f} {resp.confidence.reasons}")
    typer.echo(f"stages {json.dumps(resp.stage_ms)}")


# --------------------------------------------------------------------------- #
@app.command()
def transcribe(
    path: Path = typer.Argument(..., exists=True, readable=True),
    language: str = typer.Option("", help="BCP-47 hint, e.g. hi-IN. Empty = auto-detect."),
) -> None:
    """Transcribe an audio file through the configured provider chain."""
    from .stt import STTRouter

    router = STTRouter()
    if not router.available:
        typer.echo("No STT provider configured. Set SARVAM_API_KEY or ELEVENLABS_API_KEY.")
        raise typer.Exit(code=1)
    typer.echo(f"provider chain: {' -> '.join(router.chain)}")

    async def _run():
        try:
            return await router.transcribe(path.read_bytes(), path.name, "", language or None)
        finally:
            await router.aclose()

    result = asyncio.run(_run())
    typer.echo(result.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
@app.command()
def ask(
    path: Path = typer.Argument(..., exists=True, readable=True),
    language: str = typer.Option(""),
) -> None:
    """Full voice path: audio file in, grounded answer out."""
    from .harness.pipeline import RAGPipeline

    pipe = RAGPipeline.load()

    async def _run():
        try:
            return await pipe.answer_voice(
                path.read_bytes(), filename=path.name, language=language or None
            )
        finally:
            await pipe.aclose()

    resp = asyncio.run(_run())
    typer.echo(resp.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
@app.command()
def bench(
    n: int = typer.Option(200, help="Measured queries (after warmup)."),
    warmup: int = typer.Option(20),
    cache: bool = typer.Option(False, help="Measure the cached path instead of cold."),
    out: Path = typer.Option(REPORT_DIR / "latency.json"),
) -> None:
    """P50/P70/P90/P95/P99/P100 latency over the query path."""
    from .eval.bench import run_benchmark

    result = asyncio.run(run_benchmark(n=n, warmup=warmup, use_cache=cache))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    typer.echo(result.to_markdown())
    typer.echo(f"\nsaved {out}")


# --------------------------------------------------------------------------- #
@app.command("eval-chunking")
def eval_chunking(
    strategies: str = typer.Option("", help="Comma-separated subset; default runs all."),
    max_queries: int = typer.Option(300),
    documents: int = typer.Option(0, help="Cap documents for a faster sweep. 0 = all."),
    out: Path = typer.Option(REPORT_DIR / "chunking.json"),
) -> None:
    """Build one index per strategy and compare quality, cost and latency."""
    from .eval.chunk_eval import evaluate_strategies, save, to_markdown
    from .ingest.loader import Corpus

    corpus = Corpus.load(CORPUS_DIR)
    if documents:
        keep = corpus.documents[:documents]
        doc_ids = {d.doc_id for d in keep}
        corpus = Corpus(
            documents=keep,
            queries=[q for q in corpus.queries if any(d in doc_ids for d in q.relevant_doc_ids)],
            stats=corpus.stats,
        )
    reports = evaluate_strategies(
        corpus, [x for x in strategies.split(",") if x] or None, max_queries=max_queries
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(reports, out)
    typer.echo(to_markdown(reports))
    typer.echo(f"\nsaved {out}")


# --------------------------------------------------------------------------- #
@app.command("eval-guardrails")
def eval_guardrails(
    in_domain: int = typer.Option(10, help="In-domain queries sampled from the corpus."),
    out: Path = typer.Option(REPORT_DIR / "guardrails.json"),
) -> None:
    """Adversarial suite: unsafe, injection, off-topic, unanswerable, plus real queries."""
    from .eval.guardrail_eval import main as run_guardrails
    from .ingest.loader import Corpus

    sample: list[str] = []
    try:
        corpus = Corpus.load(CORPUS_DIR)
        sample = [q.query for q in corpus.queries if q.relevant_doc_ids][:in_domain]
    except FileNotFoundError:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_guardrails(out=out, in_domain=sample)
    typer.echo(report.to_markdown())
    typer.echo(f"\nsaved {out}")


# --------------------------------------------------------------------------- #
@app.command()
def report(
    reports_dir: Path = typer.Option(REPORT_DIR),
    target: Path = typer.Option(REPO_ROOT / "BENCHMARKS.md"),
) -> None:
    """Regenerate BENCHMARKS.md from whatever reports exist. Tables are never hand-edited."""
    from .eval.report import render

    render(reports_dir, target)
    typer.echo(f"wrote {target}")


@app.command()
def strategies() -> None:
    """List available chunking strategies."""
    from .chunking import list_strategies

    for name in list_strategies():
        typer.echo(name)


@app.command("index-info")
def index_info() -> None:
    """Show index metadata, including index-build chunking cost."""
    s = get_settings()
    meta = Path(s.index_dir) / "meta.json"
    if not meta.exists():
        typer.echo("no index; run `vrag ingest`")
        raise typer.Exit(code=1)
    typer.echo(meta.read_text())


@app.command()
def serve(
    host: str = typer.Option(""),
    port: int = typer.Option(0),
    reload: bool = typer.Option(False),
) -> None:
    """Run the HTTP API and the browser demo."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "vrag.api:app",
        host=host or s.host,
        port=port or s.port,
        reload=reload,
        log_config=None,
    )


if __name__ == "__main__":
    app()
