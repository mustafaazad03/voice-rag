---
title: Voice RAG (Sarvam + hybrid retrieval)
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Voice-in, grounded-answer-out RAG over MSMARCO-XI in ~22 ms
---

# Voice-Enabled RAG

Speak a question, get an answer grounded in retrieved passages — or an explicit
refusal when the corpus cannot support one.

**Pipeline:** voice → Sarvam STT → hybrid retrieval (HNSW + BM25, RRF fusion,
feature reranker) → extractive generation → grounding check.

Query path runs at **P50 22 ms / P100 63 ms**, entirely inside the 200 ms budget;
speech-to-text is a third-party network call and is reported separately.

Hold **Space** to talk, or type a question.

Full write-up, benchmarks and the chunking-strategy comparison are in the
[source repository](https://github.com/). See `BENCHMARKS.md` there.
