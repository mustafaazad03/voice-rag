"""Step 01 — ingest ai4bharat/MSMARCO-XI into normalized documents + a gold eval set.

Row shape (config "default"):
    target_lang, query, Eng_Query, Answer, Eng_Answer, query_id, query_type,
    passages{English_passages[], Translated_passages[], is_selected[]}

The dataset is 55 GB, so ingestion always streams and stops at `limit_queries`.
`is_selected` gives us free relevance labels — they become the eval qrels and the
training signal for the stage-2 reranker.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..models import Document
from ..obs import get_logger
from .normalize import Deduplicator, clean_text

log = get_logger("ingest.loader")

DATASET = "ai4bharat/MSMARCO-XI"
MIN_PASSAGE_CHARS = 80


@dataclass(frozen=True)
class EvalQuery:
    query_id: int
    query: str
    lang: str
    answer: str
    relevant_doc_ids: list[str]
    all_doc_ids: list[str]
    query_type: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


@dataclass(frozen=True)
class Corpus:
    documents: list[Document]
    queries: list[EvalQuery]
    stats: dict

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "documents.jsonl").open("w", encoding="utf-8") as fh:
            for d in self.documents:
                fh.write(d.model_dump_json() + "\n")
        with (path / "queries.jsonl").open("w", encoding="utf-8") as fh:
            for q in self.queries:
                fh.write(q.to_json() + "\n")
        (path / "ingest_stats.json").write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: Path) -> Corpus:
        docs = [
            Document.model_validate_json(line)
            for line in (path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        queries = [
            EvalQuery(**json.loads(line))
            for line in (path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        stats_path = path / "ingest_stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        return cls(documents=docs, queries=queries, stats=stats)


def load_corpus(
    *,
    limit_queries: int = 4000,
    langs: Iterable[str] | None = None,
    split: str = "validation",
    text_field: str = "translated",
    dedupe_distance: int = 12,
    seed_rows: list[dict] | None = None,
) -> Corpus:
    """Stream the dataset and build the corpus.

    text_field: "translated" (Indic), "english", or "both" (indexes each side as a
    separate document — useful for code-switched voice queries).
    """
    wanted = {lang_code(x) for x in langs} if langs else None
    rows = seed_rows if seed_rows is not None else _stream(split)

    dedup = Deduplicator(max_distance=dedupe_distance)
    documents: list[Document] = []
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    skipped_short = 0
    scanned = 0

    for row in rows:
        scanned += 1
        lang = lang_code(row.get("target_lang") or "unknown")
        if wanted and lang not in wanted:
            continue

        passages = row.get("passages") or {}
        variants = _variants(passages, text_field)
        if not variants:
            continue

        selected = list(passages.get("is_selected") or [])
        qid = int(row.get("query_id") or scanned)
        rel: list[str] = []
        all_ids: list[str] = []

        for tag, texts in variants:
            doc_lang = "eng" if tag == "en" else lang
            for i, raw in enumerate(texts):
                text = clean_text(raw)
                if len(text) < MIN_PASSAGE_CHARS:
                    skipped_short += 1
                    continue
                doc_id = f"{qid}:{tag}:{i}"
                if doc_id in seen_ids or dedup.is_duplicate(text):
                    continue
                seen_ids.add(doc_id)
                is_sel = bool(selected[i]) if i < len(selected) else False
                all_ids.append(doc_id)
                if is_sel:
                    rel.append(doc_id)
                documents.append(
                    Document(
                        doc_id=doc_id,
                        text=text,
                        lang=doc_lang,
                        query_id=qid,
                        query_type=row.get("query_type"),
                        is_selected=is_sel,
                        # Selected passages are human-judged answer-bearing: a real
                        # trust prior rather than a made-up constant (step 04).
                        trust=0.85 if is_sel else 0.5,
                        meta={"passage_index": i, "variant": tag},
                    )
                )

        query_text = clean_text(
            row.get("query") if text_field != "english" else row.get("Eng_Query")
        ) or clean_text(row.get("Eng_Query"))
        answer = clean_text(row.get("Answer") or row.get("Eng_Answer") or "")
        if query_text and all_ids:
            queries.append(
                EvalQuery(
                    query_id=qid,
                    query=query_text,
                    lang=lang,
                    answer=answer,
                    relevant_doc_ids=rel,
                    all_doc_ids=all_ids,
                    query_type=row.get("query_type"),
                )
            )
        if len(queries) >= limit_queries:
            break

    stats = {
        "rows_scanned": scanned,
        "documents": len(documents),
        "queries": len(queries),
        "skipped_short": skipped_short,
        "languages": sorted({d.lang for d in documents}),
        **dedup.stats,
    }
    log.info("ingest_complete", **{k: v for k, v in stats.items() if k != "languages"})
    return Corpus(documents=documents, queries=queries, stats=stats)


def _variants(passages: dict, text_field: str) -> list[tuple[str, list[str]]]:
    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    if text_field == "english":
        return [("en", english)] if english else []
    if text_field == "both":
        return [(t, p) for t, p in (("xx", translated), ("en", english)) if p]
    return [("xx", translated)] if translated else []


def _stream(split: str) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install 'vrag[ingest]' to pull the dataset") from exc
    log.info("streaming_dataset", dataset=DATASET, split=split)
    return iter(load_dataset(DATASET, split=split, streaming=True))


def lang_code(value: str) -> str:
    """'asm_Beng' -> 'asm'; 'hi' -> 'hi'. Keeps filters readable."""
    return (value or "unknown").split("_")[0].lower()
