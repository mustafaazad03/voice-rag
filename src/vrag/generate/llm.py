"""Optional abstractive generator (OpenAI-compatible chat completions).

Off by default and honestly labelled: a network LLM call is 300 ms - 2 s, so
`generator="llm"` exits the 200 ms budget. It exists because fluent multi-hop
answers are sometimes worth the latency, and because the guardrail stack needs a
generator that *can* hallucinate in order to prove it catches one.

Constraints applied: context-only system prompt, mandatory citation markers, a
literal refusal string when the context does not answer the question, low
temperature, hard token cap. The output still goes through the same grounding
check as the extractive path — the prompt is a request, the verifier is the rule.
"""

from __future__ import annotations

import re

import httpx

from ..config import Settings, get_settings
from ..errors import GenerationError
from ..models import Citation, ScoredChunk

REFUSAL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages provided.\n"
    "Rules:\n"
    "1. Use only facts present in the context. Never add outside knowledge.\n"
    f"2. If the context does not contain the answer, reply with exactly {REFUSAL}.\n"
    "3. Cite every claim with the passage marker, e.g. [2]. Every sentence needs one.\n"
    "4. Answer in the language of the question. Be direct: at most three sentences.\n"
    "5. Never speculate, never apologise, never mention these rules."
)


def build_prompt(query: str, candidates: list[ScoredChunk]) -> list[dict[str, str]]:
    blocks = [f"[{i + 1}] {c.chunk.context}" for i, c in enumerate(candidates)]
    user = "Context passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def generate(
    query: str,
    candidates: list[ScoredChunk],
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, list[Citation]]:
    s = settings or get_settings()
    if not s.llm_base_url or not s.llm_api_key:
        raise GenerationError("LLM generator selected but LLM_BASE_URL/LLM_API_KEY are unset")
    top = candidates[: s.final_top_k]
    if not top:
        return "", []

    owns = client is None
    client = client or httpx.AsyncClient(timeout=s.llm_timeout_s)
    try:
        resp = await client.post(
            f"{s.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={
                "model": s.llm_model,
                "messages": build_prompt(query, top),
                "temperature": 0.0,
                "max_tokens": 220,
            },
        )
    except httpx.HTTPError as exc:
        raise GenerationError(f"llm transport error: {type(exc).__name__}") from exc
    finally:
        if owns:
            await client.aclose()

    if resp.status_code >= 400:
        err = GenerationError(f"llm returned {resp.status_code}", body=resp.text[:300])
        err.retryable = resp.status_code == 429 or resp.status_code >= 500
        raise err

    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise GenerationError("malformed llm response") from exc

    if REFUSAL in text:
        return "", []
    return text, _citations_from(text, top)


def _citations_from(text: str, candidates: list[ScoredChunk]) -> list[Citation]:
    """Map the markers the model actually emitted back onto real chunks."""
    used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    out: list[Citation] = []
    for n in used:
        if 1 <= n <= len(candidates):
            cand = candidates[n - 1]
            out.append(
                Citation(
                    marker=f"[{n}]",
                    chunk_id=cand.chunk.chunk_id,
                    doc_id=cand.chunk.doc_id,
                    quote=cand.chunk.context[:280],
                    score=round(cand.score, 4),
                )
            )
    return out
