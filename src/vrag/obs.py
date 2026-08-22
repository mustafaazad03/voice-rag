"""Observability: structured logs, per-stage spans, in-process metrics (step 10)."""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import structlog

from .config import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    s = get_settings()
    logging.basicConfig(format="%(message)s", level=getattr(logging, s.log_level.upper(), 20))
    renderer = (
        structlog.processors.JSONRenderer() if s.log_json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, s.log_level.upper(), 20)
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- #
# Span timing
# --------------------------------------------------------------------------- #


@dataclass
class Trace:
    """Collects wall-clock per stage. `elapsed_ms` drives the latency budget."""

    trace_id: str = field(default_factory=new_trace_id)
    t0: float = field(default_factory=time.perf_counter)
    spans: dict[str, float] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def remaining_ms(self, budget_ms: float) -> float:
        return budget_ms - self.elapsed_ms

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spans[name] = self.spans.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def note(self, **kw: object) -> None:
        self.notes.update(kw)

    def rounded(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.spans.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class Metrics:
    """Tiny thread-safe registry. Prometheus text exposition, no client dep."""

    def __init__(self, window: int = 5000) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._hist[name].append(value)

    def percentiles(
        self, name: str, ps: tuple[float, ...] = (50, 70, 90, 95, 99, 100)
    ) -> dict[str, float]:
        with self._lock:
            data = sorted(self._hist.get(name, ()))
        return percentiles(data, ps)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = {
                (n if not lb else f"{n}{{{','.join(f'{k}={v}' for k, v in lb)}}}"): v
                for (n, lb), v in self._counters.items()
            }
            hists = {n: len(d) for n, d in self._hist.items()}
        return {"counters": counters, "histogram_counts": hists}

    def prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = list(self._counters.items())
            hist_names = list(self._hist.keys())
        for (name, labels), value in sorted(counters):
            lb = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f"vrag_{name}{{{lb}}} {value}" if lb else f"vrag_{name} {value}")
        for name in sorted(hist_names):
            pcts = self.percentiles(name)
            for p, v in pcts.items():
                lines.append(f'vrag_{name}_ms{{quantile="{p}"}} {v:.3f}')
        return "\n".join(lines) + "\n"


def percentiles(
    data: list[float], ps: tuple[float, ...] = (50, 70, 90, 95, 99, 100)
) -> dict[str, float]:
    """Nearest-rank percentiles. P100 == max, which is what the brief asks for."""
    if not data:
        return {str(int(p)): 0.0 for p in ps}
    ordered = sorted(data)
    out: dict[str, float] = {}
    for p in ps:
        rank = max(1, math.ceil(p / 100.0 * len(ordered)))
        out[str(int(p))] = round(ordered[min(rank, len(ordered)) - 1], 3)
    return out


METRICS = Metrics()
