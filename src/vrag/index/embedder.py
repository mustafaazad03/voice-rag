"""ONNX sentence embedder.

Torch-free on purpose: onnxruntime + a `tokenizers` fast tokenizer keeps query-time
encode in the single-digit milliseconds, which is what the 200ms budget needs.

Exposes token-level hidden states as well as pooled vectors, because the semantic
and late-chunking strategies need them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from ..config import Settings, get_settings
from ..obs import get_logger

log = get_logger("index.embedder")


@dataclass(frozen=True)
class TokenizedBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    offsets: list[list[tuple[int, int]]]  # char spans, for late chunking


class Embedder:
    """Thread-safe: ORT sessions are, the tokenizer is guarded by a lock."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        model_path = hf_hub_download(self._s.embed_repo, self._s.embed_onnx_file)
        tok_path = hf_hub_download(self._s.embed_repo, self._s.embed_tokenizer_file)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._s.onnx_intra_threads
        opts.inter_op_num_threads = self._s.onnx_inter_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._sess.get_inputs()}

        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(self._s.embed_max_tokens)
        self._tok.enable_padding(pad_id=1, pad_token="<pad>", pad_type_id=0)
        self._tok_lock = threading.Lock()
        self.dim = self._s.embed_dim

    # -- tokenization ------------------------------------------------------
    def tokenize(self, texts: list[str]) -> TokenizedBatch:
        with self._tok_lock:
            encs = self._tok.encode_batch(texts)
        return TokenizedBatch(
            input_ids=np.array([e.ids for e in encs], dtype=np.int64),
            attention_mask=np.array([e.attention_mask for e in encs], dtype=np.int64),
            offsets=[list(e.offsets) for e in encs],
        )

    # -- forward -----------------------------------------------------------
    def _forward(self, batch: TokenizedBatch) -> np.ndarray:
        feeds = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(batch.input_ids)
        return self._sess.run(None, {k: v for k, v in feeds.items() if k in self._input_names})[0]

    def forward_ids(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Escape hatch for strategies that tokenize with their own settings."""
        return self._forward(TokenizedBatch(input_ids, attention_mask, []))

    def token_embeddings(self, texts: list[str]) -> tuple[np.ndarray, TokenizedBatch]:
        """(batch, seq, dim) last hidden states plus the tokenization that produced them."""
        batch = self.tokenize(texts)
        return self._forward(batch), batch

    # -- pooled ------------------------------------------------------------
    def encode(self, texts: list[str], *, prefix: str = "", batch_size: int = 64) -> np.ndarray:
        """Batched encode with length bucketing.

        Padding is to the longest item in the batch, so mixing a 20-token chunk
        with a 190-token one makes the short one cost as much as the long one.
        Sorting by length first and scattering the results back measured ~2.5x
        throughput on the MS MARCO-XI slice — index builds go from tens of minutes
        to a few. Order of the returned array still matches the input.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = self.tokenize([prefix + texts[i] for i in idx])
            out[idx] = mean_pool(self._forward(batch), batch.attention_mask)
        return l2_normalize(out)

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(texts, prefix=self._s.embed_passage_prefix, batch_size=batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], prefix=self._s.embed_query_prefix)[0]


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    log.info("loading_embedder", repo=get_settings().embed_repo)
    return Embedder()
