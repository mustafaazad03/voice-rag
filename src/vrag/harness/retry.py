"""Reliability primitives for external calls: retry, backoff, circuit breaker."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ..errors import CircuitOpen, VRagError
from ..obs import METRICS, get_logger

log = get_logger("harness.retry")
T = TypeVar("T")


@dataclass
class CircuitBreaker:
    """Stops hammering a provider that is already down.

    closed -> (failures >= threshold) -> open -> (after reset_s) -> half-open
    A single success in half-open closes it again; a failure re-opens it.
    """

    name: str
    failure_threshold: int = 4
    reset_s: float = 20.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if (time.monotonic() - self._opened_at) >= self.reset_s else "open"

    def check(self) -> None:
        if self.state == "open":
            METRICS.inc("circuit_rejected_total", provider=self.name)
            raise CircuitOpen(f"{self.name} circuit is open", provider=self.name)

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            METRICS.inc("circuit_opened_total", provider=self.name)
            log.warning("circuit_open", provider=self.name, failures=self._failures)


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    timeout_s: float | None = None,
    base_delay_s: float = 0.12,
    max_delay_s: float = 1.0,
    breaker: CircuitBreaker | None = None,
    label: str = "call",
) -> T:
    """Run `fn` with timeout, bounded retries and full-jitter backoff.

    Only errors marked `retryable` (plus timeouts) are retried; a 4xx-style
    failure fails fast instead of burning the latency budget.
    """
    if breaker:
        breaker.check()

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = await (asyncio.wait_for(fn(), timeout_s) if timeout_s else fn())
        except TimeoutError as exc:
            last = exc
            METRICS.inc("call_timeout_total", label=label)
        except VRagError as exc:
            last = exc
            METRICS.inc("call_error_total", label=label, code=exc.code)
            if not exc.retryable:
                if breaker:
                    breaker.record_failure()
                raise
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise anything
            last = exc
            METRICS.inc("call_error_total", label=label, code=type(exc).__name__)
        else:
            if breaker:
                breaker.record_success()
            return result

        if breaker:
            breaker.record_failure()
        if attempt < attempts:
            delay = random.uniform(0, min(max_delay_s, base_delay_s * (2 ** (attempt - 1))))
            log.info("retrying", label=label, attempt=attempt, delay_s=round(delay, 3))
            await asyncio.sleep(delay)

    assert last is not None
    raise last
