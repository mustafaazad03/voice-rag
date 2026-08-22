"""Render BENCHMARKS.md from the JSON reports.

Keeps the document and the measurements from drifting apart: the tables are
generated, never hand-edited, so a stale number is impossible rather than
merely unlikely.
"""

from __future__ import annotations

import json
from pathlib import Path

MARKERS = {
    "latency": "<!-- BENCH:LATENCY -->",
    "chunking": "<!-- BENCH:CHUNKING -->",
    "guardrails": "<!-- BENCH:GUARDRAILS -->",
}
PERCENTILES = (50, 70, 90, 95, 99, 100)


def latency_section(data: dict, warm: dict | None = None) -> str:
    lat = data["latency_ms"]
    lines = [
        MARKERS["latency"],
        "",
        "## 1 · Latency",
        "",
        f"`vrag bench --n {data['queries']} --warmup {data['warmup']}` — queries sampled "
        "from the ingested corpus, cache cleared before every call, warm-up discarded. "
        "Percentiles are nearest-rank.",
        "",
        "**Query path — guardrails → embed → hybrid retrieval → merge → rerank → "
        "confidence → generate → grounding check**",
        "",
        "| P50 | P70 | P90 | P95 | P99 | P100 | mean | under 200 ms |",
        "|---|---|---|---|---|---|---|---|",
        "| "
        + " | ".join(f"**{lat[str(p)]:.1f} ms**" for p in PERCENTILES)
        + f" | {data['mean_ms']:.1f} ms | {data['under_target'] * 100:.1f}% |",
        "",
        f"Index: {data['index']['size']:,} chunks from {data['index']['documents']:,} passages, "
        f"strategy `{data['index']['strategy']}`.",
        "",
        "### Where the time goes",
        "",
        "| stage | P50 | P95 | P100 |",
        "|---|---|---|---|",
    ]
    for stage, p in sorted(data["stage_ms"].items(), key=lambda kv: -kv[1]["50"]):
        lines.append(f"| `{stage}` | {p['50']:.2f} ms | {p['95']:.2f} ms | {p['100']:.2f} ms |")

    lines += [
        "",
        "The query encoder dominates and is independent of corpus size; retrieval over "
        "the HNSW + BM25 pair is a few milliseconds and grows logarithmically. That is why "
        "the budget holds as the index grows — the constant term is the model, not the data.",
        "",
        f"Outcome mix over the run: `{json.dumps(data['status_counts'])}`. Abstentions are "
        "the guardrails working, not failures — see section 3.",
    ]

    if warm:
        wl = warm["latency_ms"]
        lines += [
            "",
            "### Cached path",
            "",
            "| P50 | P70 | P100 |",
            "|---|---|---|",
            "| " + " | ".join(f"{wl[str(p)]:.2f} ms" for p in (50, 70, 100)) + " |",
            "",
            "Exact and semantic cache hits skip retrieval entirely; the semantic layer is "
            "checked straight after the query embedding, so a paraphrase costs the encoder "
            "and nothing else.",
        ]
    return "\n".join(lines)


def chunking_section(reports: list[dict]) -> str:
    lines = [
        MARKERS["chunking"],
        "",
        "## 2 · Chunking strategies",
        "",
        "`vrag eval-chunking` — one real index built per strategy over the same passages, "
        "same queries, same reranker. Relevance labels come from the dataset's `is_selected` "
        "field. Retrieval latency excludes generation, so it is comparable across rows.",
        "",
        "| strategy | chunks | mean chars | recall@5 | recall@10 | hit@5 | MRR@10 | nDCG@10 | retrieve P50 | build |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda r: -r.get("ndcg_at_10", 0)):
        if r.get("error"):
            lines.append(f"| `{r['strategy']}` | — | — | — | — | — | — | — | — | {r['error']} |")
            continue
        lines.append(
            f"| `{r['strategy']}` | {r['chunks']:,} | {r['mean_chunk_chars']:.0f} | "
            f"{r['recall_at_5']:.3f} | {r['recall_at_10']:.3f} | {r['hit_at_5']:.3f} | "
            f"{r['mrr_at_10']:.3f} | {r['ndcg_at_10']:.3f} | {r['query_ms']['50']:.2f} ms | "
            f"{r['build_ms'].get('total', 0) / 1000:.0f} s |"
        )
    return "\n".join(lines)


def guardrail_section(report: dict) -> str:
    lines = [
        MARKERS["guardrails"],
        "",
        "## 3 · Guardrails",
        "",
        "`vrag eval-guardrails` — adversarial cases plus real in-domain queries sampled from "
        "the corpus, scored in both directions.",
        "",
        f"- accuracy **{report['accuracy']:.1%}**",
        f"- under-refusals (answered something it should not have): **{report['under_refusal']}**",
        f"- over-refusals (refused something it could answer): **{report['over_refusal']}**",
        f"- in-domain answer rate: **{report['in_domain_answer_rate']:.1%}**",
        f"- answers passing the grounding check: **{report['grounded_rate']:.1%}**",
        "",
        "| category | expected | outcome | correct |",
        "|---|---|---|---|",
    ]
    for c in report["cases"]:
        lines.append(
            f"| {c['category']} | {c['expected']} | `{c['status']}` | "
            f"{'yes' if c['correct'] else '**no**'} |"
        )
    return "\n".join(lines)


def render(reports_dir: Path, target: Path) -> None:
    text = target.read_text()
    sections: dict[str, str] = {}

    latency = _load(reports_dir / "latency.json")
    if latency:
        sections["latency"] = latency_section(latency, _load(reports_dir / "latency_warm.json"))
    chunking = _load(reports_dir / "chunking.json")
    if chunking:
        sections["chunking"] = chunking_section(chunking)
    guardrails = _load(reports_dir / "guardrails.json")
    if guardrails:
        sections["guardrails"] = guardrail_section(guardrails)

    for key, marker in MARKERS.items():
        if key not in sections:
            continue
        start = text.index(marker)
        end = _next_marker(text, start + len(marker))
        text = text[:start] + sections[key] + "\n\n" + text[end:]
    target.write_text(text.rstrip() + "\n")


def _next_marker(text: str, cursor: int) -> int:
    positions = [text.index(m, cursor) for m in MARKERS.values() if m in text[cursor:]]
    return min(positions) if positions else len(text)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None
