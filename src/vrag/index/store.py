"""Index assembly and persistence: chunks + dense + sparse + corpus statistics."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..chunking import get_chunker
from ..config import Settings, get_settings
from ..errors import IndexNotReady
from ..models import Chunk, Document
from ..obs import get_logger
from .dense import DenseIndex
from .embedder import Embedder, get_embedder
from .sparse import SparseIndex

log = get_logger("index.store")
META_FILE = "meta.json"


@dataclass
class ChunkStore:
    chunks: list[Chunk]
    dense: DenseIndex
    sparse: SparseIndex
    centroid: np.ndarray  # corpus mean vector: the off-topic reference point
    meta: dict

    @property
    def size(self) -> int:
        return len(self.chunks)

    def by_index(self, i: int) -> Chunk:
        return self.chunks[i]

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        documents: list[Document],
        *,
        strategy: str | None = None,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        chunk_params: dict | None = None,
    ) -> ChunkStore:
        s = settings or get_settings()
        strategy = strategy or s.chunk_strategy
        emb = embedder or get_embedder()
        t0 = time.perf_counter()

        chunker = get_chunker(strategy, **(chunk_params or {}))
        out = chunker.chunk_documents(documents)
        chunks = [c for c in out.chunks if c.text.strip()]
        if not chunks:
            raise IndexNotReady("chunking produced no chunks", strategy=strategy)
        chunk_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if out.embeddings is not None and len(out.embeddings) == len(out.chunks):
            # Late chunking already produced context-aware vectors.
            keep = [i for i, c in enumerate(out.chunks) if c.text.strip()]
            vectors = out.embeddings[keep].astype(np.float32)
        else:
            vectors = emb.encode_passages([c.indexable for c in chunks])
        embed_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        dense = DenseIndex.build(
            vectors, m=s.hnsw_m, ef_construction=s.hnsw_ef_construction, ef_search=s.hnsw_ef_search
        )
        sparse = SparseIndex.build([c.indexable for c in chunks])
        index_ms = (time.perf_counter() - t2) * 1000

        centroid = vectors.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

        meta = {
            "strategy": strategy,
            "chunk_params": chunk_params or {},
            "dim": int(vectors.shape[1]),
            "size": len(chunks),
            "documents": len(documents),
            "embed_repo": s.embed_repo,
            "build_ms": {
                "chunk": round(chunk_ms, 1),
                "embed": round(embed_ms, 1),
                "index": round(index_ms, 1),
            },
            "chunk_chars": {
                "mean": round(float(np.mean([len(c.text) for c in chunks])), 1),
                "p50": int(np.percentile([len(c.text) for c in chunks], 50)),
                "p95": int(np.percentile([len(c.text) for c in chunks], 95)),
            },
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        log.info("index_built", strategy=strategy, chunks=len(chunks), **meta["build_ms"])
        return cls(chunks=chunks, dense=dense, sparse=sparse, centroid=centroid, meta=meta)

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(c.model_dump_json() + "\n")
        self.dense.save(path / "dense.hnsw")
        self.sparse.save(path / "sparse")
        np.save(path / "centroid.npy", self.centroid)
        (path / META_FILE).write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))
        log.info("index_saved", path=str(path), chunks=self.size)

    @classmethod
    def load(cls, path: Path, settings: Settings | None = None) -> ChunkStore:
        s = settings or get_settings()
        meta_path = path / META_FILE
        if not meta_path.exists():
            raise IndexNotReady(f"No index at {path}. Run `vrag ingest` first.", path=str(path))
        meta = json.loads(meta_path.read_text())
        chunks = [
            Chunk.model_validate_json(line)
            for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        dense = DenseIndex.load(path / "dense.hnsw", meta["dim"], len(chunks), s.hnsw_ef_search)
        sparse = SparseIndex.load(path / "sparse", len(chunks))
        centroid = np.load(path / "centroid.npy")
        return cls(chunks=chunks, dense=dense, sparse=sparse, centroid=centroid, meta=meta)
