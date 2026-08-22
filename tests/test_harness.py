from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vrag.cache import ResponseCache, normalize_key
from vrag.errors import CircuitOpen, STTProviderError
from vrag.harness.retry import CircuitBreaker, with_retry
from vrag.models import AnswerStatus, RAGResponse


# --- retry ------------------------------------------------------------------ #
async def test_retry_succeeds_after_a_transient_failure():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise STTProviderError("boom")  # retryable=True
        return "ok"

    assert await with_retry(flaky, attempts=3, base_delay_s=0) == "ok"
    assert calls["n"] == 2


async def test_non_retryable_error_fails_fast():
    calls = {"n": 0}

    async def hard_fail():
        calls["n"] += 1
        err = STTProviderError("bad request")
        err.retryable = False
        raise err

    with pytest.raises(STTProviderError):
        await with_retry(hard_fail, attempts=3, base_delay_s=0)
    assert calls["n"] == 1


async def test_retry_honours_the_timeout():
    async def slow():
        await asyncio.sleep(0.5)

    with pytest.raises(asyncio.TimeoutError):
        await with_retry(slow, attempts=1, timeout_s=0.02)


# --- circuit breaker --------------------------------------------------------- #
async def test_breaker_opens_after_repeated_failures_and_rejects_immediately():
    breaker = CircuitBreaker("test", failure_threshold=2, reset_s=60)

    async def always_fail():
        raise STTProviderError("down")

    for _ in range(2):
        with pytest.raises(STTProviderError):
            await with_retry(always_fail, attempts=1, base_delay_s=0, breaker=breaker)

    assert breaker.state == "open"
    with pytest.raises(CircuitOpen):
        await with_retry(always_fail, attempts=1, breaker=breaker)


async def test_breaker_closes_after_a_success():
    breaker = CircuitBreaker("test", failure_threshold=5)
    breaker.record_failure()
    breaker.record_success()
    assert breaker.state == "closed"


# --- cache -------------------------------------------------------------------- #
def _response(answer: str = "hello") -> RAGResponse:
    return RAGResponse(trace_id="t", status=AnswerStatus.ANSWERED, answer=answer)


def test_normalize_key_ignores_case_punctuation_and_spacing():
    assert normalize_key("What IS a  corporation??") == normalize_key("what is a corporation")


def test_exact_cache_roundtrip_marks_the_hit():
    cache = ResponseCache()
    cache.put("k", _response())
    hit = cache.get("k")
    assert hit is not None and hit.cached is True
    assert cache.get("other") is None


def test_semantic_cache_hits_a_near_identical_vector():
    cache = ResponseCache()
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.put("k", _response("cached answer"), vec)

    close = np.array([0.999, 0.045, 0.0], dtype=np.float32)
    close /= np.linalg.norm(close)
    assert cache.get_semantic(close) is not None
    assert cache.get_semantic(np.array([0.0, 1.0, 0.0], dtype=np.float32)) is None


def test_cache_clear_empties_both_layers():
    cache = ResponseCache()
    cache.put("k", _response(), np.array([1.0, 0.0], dtype=np.float32))
    cache.clear()
    assert cache.stats == {"exact_entries": 0, "semantic_entries": 0}


# --- import hygiene ---------------------------------------------------------- #
def test_every_module_imports_standalone():
    """Guards against circular imports that only appear in a particular load order.

    `vrag.stt` needs `harness.retry`, and importing a submodule executes the
    package `__init__` — so anything eager in `vrag/harness/__init__.py` that
    reaches back into `vrag.stt` breaks `import vrag.stt` as a first import.
    """
    import subprocess
    import sys

    modules = [
        "vrag.stt",
        "vrag.harness.pipeline",
        "vrag.harness.retry",
        "vrag.api",
        "vrag.cli",
        "vrag.chunking",
        "vrag.index.store",
        "vrag.retrieve.hybrid",
        "vrag.eval.bench",
        "vrag.eval.report",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"
