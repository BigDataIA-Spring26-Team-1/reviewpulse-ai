"""Guardrails for review-grounded API queries."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


PROMPT_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore (all )?(previous|prior|system|developer) instructions\b",
        re.IGNORECASE,
    ),
    re.compile(r"\breveal (the )?(system|developer) prompt\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b|\bDAN mode\b", re.IGNORECASE),
)
SECRET_EXTRACTION_PATTERNS = (
    re.compile(r"\b(api[_ -]?key|password|secret|token|credential)s?\b", re.IGNORECASE),
    re.compile(r"\b.env\b|\benvironment variables?\b", re.IGNORECASE),
)
MANIPULATION_PATTERNS = (
    re.compile(
        r"\b(write|generate|fabricate|create)\b.*\bfake\b.*\breviews?\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(hide|remove|delete|suppress)\b.*\bnegative\b.*\breviews?\b", re.IGNORECASE
    ),
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    allowed: bool
    action: str
    reason: str
    flags: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


def _confidence_for_query(query: str) -> float:
    tokens = WORD_RE.findall(query)
    if not tokens:
        return 0.0
    if len(tokens) == 1:
        return 0.35
    if len(query.strip()) > 1200:
        return 0.35
    if len(tokens) <= 3:
        return 0.6
    return 0.9


def _collect_pattern_flags(query: str) -> list[str]:
    flags: list[str] = []
    if any(pattern.search(query) for pattern in PROMPT_INJECTION_PATTERNS):
        flags.append("prompt_injection")
    if any(pattern.search(query) for pattern in SECRET_EXTRACTION_PATTERNS):
        flags.append("secret_extraction")
    if any(pattern.search(query) for pattern in MANIPULATION_PATTERNS):
        flags.append("review_manipulation")
    return flags


def evaluate_query(
    query: str,
    *,
    confidence_threshold: float = 0.5,
    hitl_enabled: bool = False,
) -> GuardrailDecision:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return GuardrailDecision(
            allowed=False,
            action="block",
            reason="Query is empty.",
            flags=("empty_query",),
            confidence=0.0,
        )

    flags = _collect_pattern_flags(normalized_query)
    blocking_flags = {"prompt_injection", "secret_extraction", "review_manipulation"}
    matched_blocking_flags = tuple(flag for flag in flags if flag in blocking_flags)
    if matched_blocking_flags:
        return GuardrailDecision(
            allowed=False,
            action="block",
            reason="Query violates review-grounded API guardrails.",
            flags=matched_blocking_flags,
            confidence=0.95,
        )

    confidence = _confidence_for_query(normalized_query)
    low_confidence = confidence < confidence_threshold
    if low_confidence and hitl_enabled:
        return GuardrailDecision(
            allowed=False,
            action="needs_human_review",
            reason="Query is too ambiguous for automated handling.",
            flags=("low_confidence_query",),
            confidence=confidence,
        )

    decision_flags = ("low_confidence_query",) if low_confidence else tuple(flags)
    return GuardrailDecision(
        allowed=True,
        action="allow",
        reason="Query passed automated guardrails.",
        flags=decision_flags,
        confidence=confidence,
    )


def build_guardrail_answer(decision: GuardrailDecision) -> str:
    if decision.action == "needs_human_review":
        return "This query needs human review before the system can answer it."
    return "I cannot answer that request because it violates the review-grounded guardrails."
