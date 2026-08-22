"""Guardrail policy tables: what to redact, what to refuse, what to say instead.

Kept as data separate from the checking logic so the policy can be reviewed and
extended without touching control flow. Patterns are intentionally narrow — a
retrieval system that refuses ordinary questions is worse than useless — and the
final safety net is the grounding check, not this list.
"""

from __future__ import annotations

import re

# --- redaction ------------------------------------------------------------- #
# Applied before the query is logged, embedded or sent anywhere.
REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("phone", re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?\d{10}(?!\w)")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)[-_][A-Za-z0-9]{16,}\b")),
]

# --- refusal categories ----------------------------------------------------- #
# Each is (rule name, pattern, user-facing message).
UNSAFE_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "self_harm",
        re.compile(
            r"\b(kill myself|suicide|end my life|self[- ]harm|hurt myself|want to die)\b", re.I
        ),
        "I can't help with this here. If you are thinking about harming yourself, please "
        "contact a local emergency number or a crisis line right now — in India, Tele-MANAS "
        "is 14416, available 24/7.",
    ),
    (
        "weapons",
        re.compile(
            r"\b(build|make|synthesi[sz]e|manufacture)\b.{0,30}\b"
            r"(bomb|explosive|nerve agent|bioweapon|ricin|sarin)\b",
            re.I,
        ),
        "I can't help with that request.",
    ),
    (
        "malware",
        re.compile(r"\b(write|create|generate)\b.{0,25}\b(ransomware|keylogger|botnet)\b", re.I),
        "I can't help with that request.",
    ),
    (
        "explicit_minor",
        re.compile(r"\b(child|minor|underage)\b.{0,20}\b(sexual|porn|nude)\b", re.I),
        "I can't help with that request.",
    ),
]

# --- prompt injection -------------------------------------------------------- #
# The query reaches an LLM prompt on the `generator="llm"` path, and reaches the
# logs on every path. Treat it as untrusted input, never as instructions.
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore (all |the )?(previous|prior|above) (instructions|rules|prompts)\b", re.I),
    re.compile(r"\b(disregard|forget) (everything|all|your) (above|before|instructions)\b", re.I),
    re.compile(r"\b(system|developer) prompt\b", re.I),
    re.compile(r"\byou are now\b|\bact as (if|though) you\b", re.I),
    re.compile(r"\breveal (your|the) (instructions|prompt|system)\b", re.I),
    re.compile(r"<\s*/?\s*(system|assistant|instructions)\s*>", re.I),
]

# --- capability boundary ------------------------------------------------------ #
# "book me a flight to Tokyo" is not off-topic — a web corpus has plenty of travel
# passages, so retrieval happily scores it above threshold and the system answers a
# question nobody asked. It is out of *capability*, not out of topic, and only an
# intent check catches it.
ACTION_REQUEST = re.compile(
    r"^\s*(please\s+)?(book|buy|order|purchase|reserve|schedule|cancel|send|email|call|text|"
    r"remind|transfer|pay|delete|install|deploy|sign\s+me)\b.{0,40}\b(me|us|my|our|for\s+me)\b",
    re.I,
)
ACTION_MESSAGE = (
    "I can only answer questions from the indexed knowledge base — I can't take actions "
    "like booking, buying or sending things."
)

INJECTION_MESSAGE = (
    "That request looks like an attempt to change how this system works rather than a "
    "question about the knowledge base. Ask a question about the indexed content instead."
)

# --- refusal copy ------------------------------------------------------------ #
INSUFFICIENT_EVIDENCE = (
    "I don't have enough supporting information in the knowledge base to answer that "
    "confidently, so I'd rather not guess."
)
OFF_TOPIC = (
    "That question falls outside the indexed knowledge base, so I can't answer it from "
    "the available sources."
)
EMPTY_QUERY = "I didn't catch a question there — could you repeat it?"
TOO_LONG = "That question is too long to process. Please shorten it."
