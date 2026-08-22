#!/usr/bin/env python
"""Fail the build when a measurement regresses.

The eval suite is only worth having if something acts on it. These are the
contract: cross one and CI goes red, so a regression has to be argued for
explicitly instead of landing quietly.

    python scripts/check_thresholds.py [reports_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# (report, json path, comparison, bound). Bounds sit slightly below current
# measurements so normal noise does not flap the build.
CHECKS: list[tuple[str, tuple[str, ...], str, float]] = [
    ("latency", ("latency_ms", "50"), "<=", 60.0),
    ("latency", ("latency_ms", "95"), "<=", 120.0),
    ("latency", ("latency_ms", "100"), "<=", 200.0),
    ("latency", ("under_target",), ">=", 1.0),
    ("guardrails", ("accuracy",), ">=", 0.75),
    ("guardrails", ("grounded_rate",), ">=", 0.95),
    ("guardrails", ("under_refusal",), "<=", 2),
]
# Retrieval quality of the shipped strategy, checked against the chunking report.
STRATEGY_MIN_NDCG = 0.40


def _dig(data: dict, path: tuple[str, ...]):
    for key in path:
        data = data[key]
    return data


def main(reports: Path) -> int:
    failures: list[str] = []
    checked = 0

    for name, path, op, bound in CHECKS:
        report = reports / f"{name}.json"
        if not report.exists():
            failures.append(f"MISSING {report} — run `make eval bench`")
            continue
        value = _dig(json.loads(report.read_text()), path)
        ok = value <= bound if op == "<=" else value >= bound
        checked += 1
        label = f"{name}.{'.'.join(path)}"
        print(f"{'ok  ' if ok else 'FAIL'} {label} = {value} (need {op} {bound})")
        if not ok:
            failures.append(f"{label} = {value}, needs {op} {bound}")

    chunking = reports / "chunking.json"
    if chunking.exists():
        rows = json.loads(chunking.read_text())
        best = max(rows, key=lambda r: r.get("ndcg_at_10") or 0)
        checked += 1
        ok = best["ndcg_at_10"] >= STRATEGY_MIN_NDCG
        print(
            f"{'ok  ' if ok else 'FAIL'} chunking.best({best['strategy']}).ndcg_at_10 = "
            f"{best['ndcg_at_10']} (need >= {STRATEGY_MIN_NDCG})"
        )
        if not ok:
            failures.append(f"best strategy nDCG@10 {best['ndcg_at_10']} < {STRATEGY_MIN_NDCG}")

    print(f"\n{checked} checks, {len(failures)} failing")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "reports")))
