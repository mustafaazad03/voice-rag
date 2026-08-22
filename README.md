# vrag — voice-enabled RAG

Speak a question, get an answer that is grounded in a retrieved passage and cited
back to it. Sarvam AI for speech-to-text with ElevenLabs as an automatic
fallback, eight chunking strategies benchmarked against each other, hybrid
BM25 + dense retrieval with a learned reranker, and a query path that finishes
well inside 200 ms.

Built on [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
— MS MARCO translated into 14 Indic languages, 11.5 M rows, 55 GB. Ingestion
streams, so you choose how much of it to index; nothing here needs the full 55 GB
on disk.

One thing worth knowing before you run it: the split is *ordered by target
language*, so a short prefix gives you one Indic language plus English rather than
a sample of all fourteen. `--langs hin,ben,tam` filters as it streams if you want
specific ones, at the cost of scanning further into the shards.

```
voice ─▶ STT ─▶ guardrails ─▶ embed ─▶ hybrid retrieval ─▶ rerank
                                             │
                     confidence ◀────────────┘
                          │
        insufficient ◀────┴────▶ constrained generation ─▶ grounding check ─▶ answer + citations
```

## What the brief asked for, and where it lives

| requirement | where | short version |
|---|---|---|
| Voice → STT | [`src/vrag/stt/`](src/vrag/stt/) · [#1](#1--speech-to-text--sarvam-with-elevenlabs-as-fallback) | Sarvam `saaras:v3`, ElevenLabs `scribe_v1` fallback, chosen by which env keys exist |
| Chunking, non-naive | [`src/vrag/chunking/`](src/vrag/chunking/) · [#2](#2--chunking--eight-strategies-chosen-by-measurement) | 8 strategies + overlap merging + dedup, compared on real metrics |
| Retrieval → answer | [`src/vrag/retrieve/`](src/vrag/retrieve/), [`generate/`](src/vrag/generate/) · [#3](#3--retrieval-and-generation) | BM25 + HNSW, RRF, learned reranker, citation-backed generation |
| < 200 ms end to end | [BENCHMARKS.md](BENCHMARKS.md) · [#4](#4--latency) | P50 ≈ 22 ms, P100 ≈ 63 ms, 100% under target, budget enforced in code |
| P50 / P70 / P100 | `vrag bench` → [BENCHMARKS.md](BENCHMARKS.md) | 300 cold queries, warm-up discarded, nearest-rank percentiles |
| Harness, not a raw prompt | [`src/vrag/harness/`](src/vrag/harness/) · [#5](#5--the-harness) | typed stages, tool router, retries, circuit breakers, budget degradation |
| Guardrails | [`src/vrag/guardrails/`](src/vrag/guardrails/) · [#6](#6--guardrails--knowing-when-not-to-answer) | input policy, confidence abstention, per-sentence grounding check, adversarial evals |

---

## Quick start

```bash
make setup                      # venv + deps + .env
```

Put at least one key in `.env` (voice is optional; the text path works without any):

```bash
SARVAM_API_KEY=...              # preferred
ELEVENLABS_API_KEY=...          # fallback, or use alone
```

```bash
make ingest                     # stream the dataset, chunk, index, train the reranker
make test                       # 111 tests
make bench                      # P50 / P70 / P100
make eval                       # chunking sweep + adversarial guardrails
make serve                      # http://localhost:8000 — mic + text demo
```

Every table in [BENCHMARKS.md](BENCHMARKS.md) is generated from `reports/*.json`
by `vrag report`, so the document cannot drift from the measurements.

Single question, no server:

```bash
.venv/bin/vrag query "what is a corporation"
```

Full voice path from an audio file:

```bash
.venv/bin/vrag ask sample.wav
```

---

## 1 · Speech-to-text — Sarvam, with ElevenLabs as fallback

Provider selection is a pure function of which keys exist in the environment. No
flags, no code changes.

| `SARVAM_API_KEY` | `ELEVENLABS_API_KEY` | behaviour |
|---|---|---|
| set | — | Sarvam |
| — | set | ElevenLabs |
| set | set | **Sarvam first**, ElevenLabs on failure |
| — | — | voice endpoints return `503`, text endpoints keep working |

Each provider gets bounded retries with full-jitter backoff and its own circuit
breaker, so a provider that is down stops being called instead of costing every
request a timeout. A 4xx fails fast; only 429/5xx and transport errors retry.
When the chain falls through to the second provider, the response says so
(`transcription.fallback_used`).

- Sarvam: `POST https://api.sarvam.ai/speech-to-text`, header `api-subscription-key`,
  model `saaras:v3` with `mode=transcribe`, auto language detection.
  ([docs](https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe))
- ElevenLabs: `POST https://api.elevenlabs.io/v1/speech-to-text`, header `xi-api-key`,
  model `scribe_v1`. BCP-47 language hints are converted to ISO-639 automatically.
  ([docs](https://elevenlabs.io/docs/api-reference/speech-to-text/convert))

Implementation: [`src/vrag/stt/`](src/vrag/stt/)

---

## 2 · Chunking — eight strategies, chosen by measurement

Chunking is where retrieval quality is won or lost, so this is a registry of
composable strategies rather than one splitter, plus a harness that builds a real
index per strategy and scores the same queries against each.

| strategy | idea | why it exists |
|---|---|---|
| `fixed_char` | fixed character windows + overlap | the baseline everything is measured against |
| `fixed_token` | fixed subword windows on real token boundaries | never splits mid-token or overflows the encoder window |
| `recursive` | paragraph → sentence → clause → word hierarchy | breaks at the most semantic boundary that fits |
| `sentence_window` | tight indexed window, wider returned context | precise matching without starving the generator |
| `semantic` | embedding-distance breakpoints | cuts where the topic changes, not at an arbitrary offset |
| `late` | document encoded first, chunk vectors pooled after | a chunk saying "it was founded in 1998" keeps what "it" was |
| `hierarchical` | small children indexed, big parent returned | **default** — small-to-big, precision plus context |
| `metadata_aware` | decorates any of the above with a context header | makes "4 to 6 weeks" findable by "how long does a passport take" |

Beyond the split itself:

- **Overlap handling.** Overlap helps recall and wrecks a top-k list — three
  slots holding the same sentence shifted by one. After fusion, candidates from
  the same document with touching spans are merged into one widened chunk,
  keeping the best score and stitching text without repeating the shared tail.
  ([`chunking/overlap.py`](src/vrag/chunking/overlap.py))
- **Script-aware sentence splitting.** Terminators include danda `।`, double
  danda `॥` and the Urdu full stop `۔`, not just ASCII. An English-only splitter
  produces one giant "sentence" per Indic passage.
- **Metadata-aware indexing.** `index_text` (what the retriever sees) is
  separate from `text` (what the user sees and what citations quote), so a
  context header improves recall without polluting the answer.
- **Ingest-time deduplication.** MS MARCO attaches the same paragraph to dozens
  of queries. Exact fingerprinting plus banded SimHash near-dedup; on a 200-query
  slice this drops ~5% of passages that would otherwise skew IDF and fill top-k
  with copies. ([`ingest/normalize.py`](src/vrag/ingest/normalize.py))

```bash
.venv/bin/vrag eval-chunking --max-queries 250
```

builds every index and prints recall@5/@10, hit@5, MRR@10, nDCG@10, retrieval
p50 and build time side by side. `hierarchical` is the default because it won that
sweep — see [BENCHMARKS.md](BENCHMARKS.md#2--chunking-strategies), not because it
sounded good.

---

## 3 · Retrieval and generation

**Hybrid (step 02).** BM25 (`bm25s`, scipy sparse) and dense ANN (`hnswlib`,
inner-product on L2-normalised vectors) run on every query and fuse with
Reciprocal Rank Fusion. RRF is rank-based, so it needs no score calibration
between two retrievers whose scores are not comparable. Weighted min-max fusion
is available via `FUSION=weighted`.

**Two-stage rerank (step 03).** A cross-encoder is the textbook second stage and
costs 50–300 ms on CPU — the entire budget for one stage. Instead the reranker is
a small gradient-boosted model over 12 cheap features the first stage already
computed (both scores, both ranks, query-term coverage, bigram hits, chunk
geometry), trained on MS MARCO-XI's own `is_selected` labels against the hard
negatives stage 1 actually retrieves. Scoring 24 candidates costs ~0.2 ms. If no
model has been trained it falls back to a fixed linear blend, so retrieval never
hard-depends on the training step.

**Confidence (step 04).** Five signals — top cosine similarity, score gap,
multi-document agreement, lexical coverage, source trust. Trust is a real prior
(`is_selected` passages are human-judged), not a constant. The diagram's
"freshness" input is deliberately absent: MS MARCO-XI carries no timestamps, so
faking it would be theatre.

**Generation (steps 05–06).** Default is *constrained extractive* generation:
MMR selection over the sentences of the retrieved context, ordered, deduplicated
and citation-marked. Every token provably comes from a retrieved chunk, which
turns the hallucination check into verification rather than hope — and it costs
~3 ms instead of a network round trip. Every sentence carries a `[n]` marker
resolving to a chunk id, a document id and the quoted span.

`GENERATOR=llm` swaps in abstractive phrasing through any OpenAI-compatible
endpoint, with a context-only system prompt and a literal `INSUFFICIENT_CONTEXT`
refusal token. It is **outside the 200 ms target** and the response says so via
`degraded: ["llm_over_budget"]`. It also goes through the same grounding check —
the prompt is a request, the verifier is the rule.

---

## 4 · Latency

Numbers come from `vrag bench`, which replays sampled real queries with the cache
cleared before every call, discards a warm-up window, and reports nearest-rank
percentiles. See [reports/latency.json](reports/latency.json) and
[BENCHMARKS.md](BENCHMARKS.md) for the current run and the machine it ran on.

Scope, stated precisely: the measured path is **input guardrails → query
embedding → hybrid retrieval → overlap merge → rerank → confidence → generation →
grounding check**, i.e. "chunking + vector DB retrieval + everything through to
final output". Chunking itself is an index-build cost, not a per-query cost — it
is reported separately by `vrag index-info` under `build_ms.chunk`.

Speech-to-text is reported as its own number and excluded from the pipeline
figure. It is a third-party network round trip; folding it in would measure the
distance to Sarvam's datacentre, not this pipeline. Every number is in the
response — `stage_ms.stt` for transcription, `latency_ms` for the query path,
`total_ms` for what the caller actually waited for — so nothing is hidden behind
a favourable definition.

The budget is enforced, not hoped for. `Trace.elapsed_ms` is checked before the
two widest stages and the pipeline degrades rather than overrunning:

| checkpoint | action when over |
|---|---|
| before retrieval, > 35% of budget | halve the candidate lists |
| before rerank, > `BUDGET_RERANK_MS` | skip rerank, keep fusion order |

Every degradation is named in `response.degraded`, so a fast answer never
silently hides that it was a cheaper answer.

Live percentiles from served traffic: `GET /api/v1/latency`. Prometheus:
`GET /metrics`.

---

## 5 · The harness

Each stage has a declared input and output type (Pydantic, `extra="forbid"`,
frozen where it should be), its own span in a trace, and its own failure policy.

- **Tool calls.** STT providers are tools behind one interface with a router that
  owns ordering, retries and breakers. The optional LLM generator is the same
  shape.
- **Retries.** Full-jitter exponential backoff, per-call timeouts, and a
  retryability flag on every error so client errors fail fast instead of burning
  budget three times.
- **Circuit breakers.** closed → open after N failures → half-open after a cooldown.
- **Structured I/O.** `QueryRequest` in, `RAGResponse` out — answer, citations,
  confidence report, grounding report, transcription, per-stage timings, cache
  status, degradation list.
- **Error recovery.** A dead reranker degrades to fusion order. A dead LLM
  degrades to the extractive generator. An ungrounded answer degrades to the
  insufficient-evidence fallback. Errors that would make the answer *wrong* are
  the only ones that surface, as `{"error": {"code", "message", "details"}}`.
- **Observability.** JSON logs with trace ids, per-stage histograms, counters,
  Prometheus exposition.
- **Caching (step 09).** Exact TTL+LRU on the normalised query, plus a semantic
  layer keyed on the query embedding at cosine ≥ 0.97 — which is what makes voice
  usable, since "what is a corporation" and "what's a corporation" are different
  strings and the same question.

Implementation: [`src/vrag/harness/`](src/vrag/harness/)

---

## 6 · Guardrails — knowing when not to answer

**Input** ([`guardrails/input_guard.py`](src/vrag/guardrails/input_guard.py),
[`policy.py`](src/vrag/guardrails/policy.py)), under 0.3 ms:

- structural rejection — empty, gibberish, over-length
- PII redaction before the query is logged, embedded or sent anywhere (email,
  card, Aadhaar, phone, API keys) — the question survives, the PII does not
- unsafe-intent refusal, with a real crisis resource for self-harm rather than a
  blank "I can't help"
- prompt-injection detection, because the query reaches an LLM prompt on the
  `llm` path and reaches the logs on every path
- capability-boundary refusal for action requests. "book me a flight to Tokyo"
  is *not* off-topic — a web corpus is full of travel passages, so retrieval
  scores it above threshold and the system cheerfully answers a question nobody
  asked. Only an intent check catches that one; the adversarial suite caught it
  first.

**Retrieval** ([`retrieve/confidence.py`](src/vrag/retrieve/confidence.py)) — two
distinct outcomes, because they deserve different answers:

- `off_topic` — top cosine below `OFF_TOPIC_SIMILARITY`; the corpus does not
  cover this subject at all
- `insufficient_evidence` — covered, but the evidence is too thin

Thresholds are calibrated from measured distributions on the default encoder
(on-topic queries score 0.90–0.95 against their passage, unrelated ones bottom
out near 0.73–0.77), not guessed. They are model-specific and the config says so.

Lexical coverage is skipped when the query and the evidence use different writing
systems. An Assamese question answered from an English passage scores exactly 0.0
coverage by construction, and on this corpus that is the *normal* case — treating
it as weak evidence turned every correct cross-lingual retrieval into a refusal
until the check became script-aware.

**Output** ([`guardrails/grounding.py`](src/vrag/guardrails/grounding.py)) —
per-sentence IDF-weighted containment against the retrieved evidence. Requiring a
*single* chunk to cover a sentence (not the union of all chunks) is what catches
the classic failure where half a fact from passage A is stitched onto half a fact
from passage B. Below `MIN_ANSWER_GROUNDING` the answer is suppressed and
replaced by the insufficient-evidence fallback — the system returns nothing
rather than something unsupported.

**Continuous evals (step 08).**

```bash
.venv/bin/vrag eval-guardrails
```

An adversarial suite covering unsafe intent, injection, off-topic, unanswerable,
empty and PII-bearing queries, mixed with real in-domain queries sampled from the
corpus. It reports **both** failure directions — under-refusal (answered
something it shouldn't) and over-refusal (refused something it could answer) —
because a suite that only measures the first is how you ship a system that
refuses everything and scores 100%.

---

## API

```
POST /api/v1/query          {"query": "...", "top_k": 5}
POST /api/v1/voice/query    multipart: audio, language?, top_k?, generator?
POST /api/v1/transcribe     multipart: audio  (STT only)
GET  /api/v1/health         index metadata, provider chain, cache stats
GET  /api/v1/config         non-secret view of the active configuration
GET  /api/v1/latency        live P50/P70/P90/P95/P99/P100
GET  /metrics               Prometheus
GET  /                      Voxcite console: push-to-talk + text
```

CORS is an explicit allowlist (never `*`). Rate limiting is a per-client token
bucket — in-process, which is honest about being per-worker; swap for Redis when
you run more than one. Swagger/ReDoc and `/openapi.json` are switched off: six
endpoints do not need a schema browser, and not shipping one is one less surface.

The console at `/` is push-to-talk: hold <kbd>Space</kbd> (or <kbd>⌥</kbd> to talk
while the field has focus), release to send. <kbd>⌘K</kbd> focuses the field,
<kbd>Esc</kbd> discards a recording without uploading it. Every answer shows its
citations, confidence, grounding score, per-stage timings and any budget
degradations that fired.

---

## Layout

```
src/vrag/
  config.py          all tunables, env-driven
  models.py          typed stage contracts
  errors.py          error taxonomy + envelope
  obs.py             logs, spans, metrics, percentiles
  harness/           pipeline orchestration, retry, circuit breaker
  stt/               sarvam, elevenlabs, router
  ingest/            dataset loader, normalisation, dedup
  chunking/          8 strategies + overlap merging
  index/             onnx embedder, hnsw, bm25, store
  retrieve/          hybrid fusion, reranker, confidence, training
  generate/          extractive (default), llm (optional)
  guardrails/        input, grounding, policy
  eval/              latency, chunking sweep, adversarial guardrails, report writer
  api.py  cli.py  cache.py
  static/            index.html + app.css + app.js (no build step)
tests/               111 tests
```

## Design decisions worth arguing about

- **Extractive generation by default.** A 200 ms budget and a network LLM call
  are mutually exclusive. Rather than quietly measuring only the retrieval half
  and calling it the pipeline, the default generator is one that actually fits,
  and the LLM path is available, honest about its cost, and held to the same
  grounding check.
- **A feature reranker instead of a cross-encoder.** Same reason. The labels to
  train it are already in the dataset.
- **Torch-free inference.** ONNX Runtime plus a fast tokenizer. Query encode is
  single-digit milliseconds and the container is small.
- **Off-topic is decided after retrieval, not before.** You cannot tell whether a
  question is outside the corpus without consulting the corpus. A pre-retrieval
  centroid check looks clever and is fragile — as the corpus broadens, the
  centroid drifts toward nothing in particular.
