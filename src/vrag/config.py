"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- speech to text -------------------------------------------------
    # Provider order is derived from which keys exist. Sarvam wins when both are set.
    sarvam_api_key: str | None = None
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"
    sarvam_mode: str = "transcribe"
    sarvam_language_code: str = "unknown"  # auto-detect

    elevenlabs_api_key: str | None = None
    elevenlabs_stt_url: str = "https://api.elevenlabs.io/v1/speech-to-text"
    elevenlabs_model: str = "scribe_v1"

    stt_timeout_s: float = 12.0
    stt_max_attempts: int = 2
    stt_max_audio_bytes: int = 20 * 1024 * 1024

    # --- embedding model -------------------------------------------------
    embed_repo: str = "intfloat/multilingual-e5-small"
    # fp32 is the default because the similarity thresholds below are calibrated
    # against it. "onnx/model_qint8_avx512_vnni.onnx" measured ~30% faster query
    # encode on the same box; switching it needs a re-index and a re-calibration
    # run (`vrag eval-guardrails`), so it is opt-in rather than a silent default.
    embed_onnx_file: str = "onnx/model.onnx"
    embed_tokenizer_file: str = "onnx/tokenizer.json"
    embed_dim: int = 384
    embed_max_tokens: int = 192
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "
    onnx_intra_threads: int = 4
    onnx_inter_threads: int = 1

    # --- index ------------------------------------------------------------
    index_dir: Path = REPO_ROOT / "data" / "index"
    chunk_strategy: str = "hierarchical"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # --- retrieval --------------------------------------------------------
    dense_top_k: int = 60
    sparse_top_k: int = 60
    rrf_k: int = 60
    rerank_top_n: int = 24
    final_top_k: int = 5
    dense_weight: float = 0.62  # used by weighted fusion mode

    fusion: Literal["rrf", "weighted"] = "rrf"
    rerank: Literal["features", "none"] = "features"

    # --- guardrails / confidence ------------------------------------------
    # Thresholds are on raw cosine similarity from the configured encoder, so they
    # are model-specific. Calibrated for intfloat/multilingual-e5-small, where an
    # on-topic query scores 0.90-0.95 against its passage and an unrelated query
    # bottoms out around 0.73-0.77. Re-measure after changing `embed_repo`:
    #   vrag eval-guardrails
    off_topic_similarity: float = 0.80  # below this, the corpus simply lacks the topic
    min_retrieval_score: float = 0.84  # below this, evidence is too weak to answer
    strong_similarity: float = 0.90  # above this, semantics alone is enough
    min_lexical_coverage: float = 0.25
    min_answer_grounding: float = 0.42
    abstain_confidence: float = 0.35
    max_query_chars: int = 512
    min_query_chars: int = 2

    # --- generation --------------------------------------------------------
    generator: Literal["extractive", "llm"] = "extractive"
    answer_max_sentences: int = 3
    answer_max_chars: int = 600

    llm_base_url: str | None = None  # any OpenAI-compatible endpoint
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 8.0

    # --- latency budget -----------------------------------------------------
    # Query-path budget (excludes third-party STT network time, reported separately).
    budget_total_ms: float = 200.0
    budget_rerank_ms: float = 130.0  # skip rerank if we are past this when it starts
    budget_generate_ms: float = 175.0

    # Run the CPU-bound retrieval block in a worker thread. True keeps the API
    # event loop responsive; False removes ~0.3ms of hand-off noise for benchmarks.
    offload_cpu: bool = True

    # --- cache ---------------------------------------------------------------
    cache_size: int = 2048
    cache_ttl_s: float = 300.0
    semantic_cache_threshold: float = 0.97

    # --- api -----------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    # NoDecode: env source would JSON-decode a list field before the validator below
    # runs, so the documented CSV form (CORS_ORIGINS=a,b) would raise. Parse it ourselves.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8000"]
    )
    rate_limit_per_min: int = 60
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def stt_providers(self) -> list[str]:
        """Ordered provider chain. Sarvam first when both keys are present."""
        chain = []
        if self.sarvam_api_key:
            chain.append("sarvam")
        if self.elevenlabs_api_key:
            chain.append("elevenlabs")
        return chain


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Test hook."""
    global _settings
    _settings = None
