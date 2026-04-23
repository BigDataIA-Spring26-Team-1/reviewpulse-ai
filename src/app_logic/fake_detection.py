"""Deterministic fake-review risk scoring for retrieved reviews."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PROMOTIONAL_PATTERNS = (
    "promo code",
    "discount code",
    "click here",
    "limited time",
    "free gift",
    "sponsored",
    "affiliate",
    "use my code",
    "guaranteed results",
)
GENERIC_SUPERLATIVES = {
    "amazing",
    "awesome",
    "best",
    "excellent",
    "fantastic",
    "incredible",
    "perfect",
    "wonderful",
}
CONCRETE_ASPECT_TERMS = {
    "battery",
    "camera",
    "screen",
    "display",
    "shipping",
    "support",
    "warranty",
    "comfort",
    "sound",
    "audio",
    "build",
    "price",
    "durability",
    "performance",
    "repair",
    "service",
}


@dataclass(frozen=True, slots=True)
class FakeDetectionResult:
    score: float
    label: str
    flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


def _clamp_score(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score_review_authenticity(
    text: str,
    *,
    threshold: float = 0.7,
) -> FakeDetectionResult:
    review_text = str(text or "").strip()
    if not review_text:
        return FakeDetectionResult(score=0.0, label="low_risk", flags=tuple())

    tokens = _tokenize(review_text)
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count if token_count else 1.0
    flags: list[str] = []
    score = 0.0

    if token_count < 8:
        flags.append("very_short_review")
        score += 0.15

    if URL_RE.search(review_text) or EMAIL_RE.search(review_text):
        flags.append("external_link_or_contact")
        score += 0.25

    lowered = review_text.lower()
    promotional_hits = [
        pattern for pattern in PROMOTIONAL_PATTERNS if pattern in lowered
    ]
    if promotional_hits:
        flags.append("promotional_language")
        score += min(0.35, 0.15 + 0.08 * len(promotional_hits))

    repeated_token_count = sum(1 for token in set(tokens) if tokens.count(token) >= 4)
    if token_count >= 10 and unique_ratio < 0.45:
        flags.append("low_vocabulary_diversity")
        score += 0.2
    if repeated_token_count:
        flags.append("repeated_terms")
        score += min(0.2, 0.08 * repeated_token_count)

    if review_text.count("!") >= 3 or review_text.count("?") >= 3:
        flags.append("excessive_punctuation")
        score += 0.1

    uppercase_letters = sum(1 for char in review_text if char.isupper())
    alpha_letters = sum(1 for char in review_text if char.isalpha())
    if alpha_letters >= 20 and uppercase_letters / alpha_letters > 0.45:
        flags.append("excessive_caps")
        score += 0.12

    superlative_count = sum(1 for token in tokens if token in GENERIC_SUPERLATIVES)
    has_concrete_aspect = any(token in CONCRETE_ASPECT_TERMS for token in tokens)
    if superlative_count >= 3 and not has_concrete_aspect:
        flags.append("generic_superlatives")
        score += 0.2

    if re.search(r"\b(5\s*stars|five\s*stars|must\s*buy|buy\s*now)\b", lowered):
        flags.append("review_bait_phrase")
        score += 0.12

    final_score = _clamp_score(score)
    suspicious_cutoff = min(threshold * 0.6, 0.55)
    if final_score >= threshold:
        label = "likely_fake"
    elif final_score >= suspicious_cutoff:
        label = "suspicious"
    else:
        label = "low_risk"

    return FakeDetectionResult(score=final_score, label=label, flags=tuple(flags))


def annotate_review(
    review: dict[str, Any],
    *,
    threshold: float = 0.7,
) -> dict[str, Any]:
    result = score_review_authenticity(
        str(review.get("review_text", "")),
        threshold=threshold,
    )
    annotated = dict(review)
    annotated["fake_review_score"] = result.score
    annotated["fake_review_label"] = result.label
    annotated["fake_review_flags"] = list(result.flags)
    return annotated


def annotate_and_filter_reviews(
    reviews: list[dict[str, Any]],
    *,
    threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], int]:
    annotated = [annotate_review(review, threshold=threshold) for review in reviews]
    kept = [
        review for review in annotated if review["fake_review_label"] != "likely_fake"
    ]
    return kept, len(annotated) - len(kept)
