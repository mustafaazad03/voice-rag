"""HTTP surface: voice + text query endpoints, health, metrics, browser demo."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .errors import IndexNotReady, RateLimited, VRagError
from .harness.pipeline import RAGPipeline
from .models import QueryRequest, RAGResponse
from .obs import METRICS, configure_logging, get_logger

log = get_logger("api")
STATIC = Path(__file__).parent / "static"

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    try:
        _state["pipeline"] = RAGPipeline.load(s)
    except IndexNotReady as exc:
        # Serve /health and a useful error instead of refusing to boot.
        log.error("index_not_ready", error=exc.message)
        _state["error"] = exc
    yield
    pipe = _state.get("pipeline")
    if isinstance(pipe, RAGPipeline):
        await pipe.aclose()


# Interactive API docs are switched off: this service exposes six endpoints and
# the README documents them. Nothing here needs a schema browser in production.
app = FastAPI(
    title="vrag",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_s = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_s.cors_origins,  # explicit allowlist, never "*"
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Brand assets for the demo page. Everything else is inlined in index.html.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --------------------------------------------------------------------------- #
# Rate limiting: in-process token bucket per client. Swap for Redis when you run
# more than one worker and actually need a shared limit.
# --------------------------------------------------------------------------- #
_buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))


def _allow(client: str) -> bool:
    s = get_settings()
    capacity = float(s.rate_limit_per_min)
    refill = capacity / 60.0
    now = time.monotonic()
    tokens, last = _buckets[client]
    tokens = min(capacity, tokens + (now - last) * refill) if last else capacity
    if tokens < 1.0:
        _buckets[client] = (tokens, now)
        return False
    _buckets[client] = (tokens - 1.0, now)
    return True


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/") and not _allow(
        request.client.host if request.client else "unknown"
    ):
        METRICS.inc("rate_limited_total")
        return JSONResponse(status_code=429, content=RateLimited("Too many requests").envelope())
    return await call_next(request)


@app.exception_handler(VRagError)
async def _vrag_error(request: Request, exc: VRagError) -> JSONResponse:
    log.warning("request_failed", path=request.url.path, code=exc.code, message=exc.message)
    METRICS.inc("http_error_total", code=exc.code)
    return JSONResponse(status_code=exc.http_status, content=exc.envelope())


def _pipeline() -> RAGPipeline:
    pipe = _state.get("pipeline")
    if not isinstance(pipe, RAGPipeline):
        raise _state.get("error") or IndexNotReady("Index not loaded. Run `vrag ingest`.")
    return pipe


# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/v1/health")
async def health() -> dict:
    pipe = _state.get("pipeline")
    s = get_settings()
    ready = isinstance(pipe, RAGPipeline)
    return {
        "status": "ok" if ready else "degraded",
        "index": pipe.store.meta if ready else None,
        "stt_chain": s.stt_providers,
        "generator": s.generator,
        "cache": pipe.cache.stats if ready else None,
    }


@app.get("/api/v1/config")
async def config() -> dict:
    """Non-secret view of the active configuration."""
    s = get_settings()
    return {
        "stt_chain": s.stt_providers,
        "embed_repo": s.embed_repo,
        "chunk_strategy": s.chunk_strategy,
        "fusion": s.fusion,
        "rerank": s.rerank,
        "generator": s.generator,
        "budget_total_ms": s.budget_total_ms,
        "final_top_k": s.final_top_k,
    }


@app.post("/api/v1/query", response_model=RAGResponse)
async def query(request: QueryRequest) -> RAGResponse:
    return await _pipeline().answer(request)


@app.post("/api/v1/voice/query", response_model=RAGResponse)
async def voice_query(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
    generator: str | None = Form(default=None),
) -> RAGResponse:
    data = await audio.read()
    base = QueryRequest(query="pending transcription", top_k=top_k, generator=generator)
    return await _pipeline().answer_voice(
        data,
        filename=audio.filename or "audio.webm",
        content_type=audio.content_type or "",
        language=language,
        request=base,
    )


@app.post("/api/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...), language: str | None = Form(default=None)
) -> dict:
    data = await audio.read()
    result = await _pipeline().stt.transcribe(
        data, audio.filename or "audio.webm", audio.content_type or "", language
    )
    return result.model_dump()


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(METRICS.prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/api/v1/latency")
async def latency() -> dict:
    """Live percentiles from served traffic, same shape as the offline benchmark."""
    return {
        "pipeline_ms": METRICS.percentiles("pipeline_latency"),
        "stt_ms": METRICS.percentiles("stt_latency"),
        "stages_ms": {
            name.removeprefix("stage_"): METRICS.percentiles(name)
            for name in METRICS.snapshot()["histogram_counts"]
            if name.startswith("stage_")
        },
    }
