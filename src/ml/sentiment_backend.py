from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
POSITIVE_LABELS = {"positive", "label_1", "4 stars", "5 stars"}
NEGATIVE_LABELS = {"negative", "label_0", "1 star", "2 stars"}
NEUTRAL_LABELS = {"neutral", "3 stars"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "brilliant",
    "comfortable",
    "durable",
    "easy",
    "excellent",
    "fantastic",
    "fast",
    "good",
    "great",
    "improved",
    "impressive",
    "incredible",
    "intuitive",
    "love",
    "perfect",
    "premium",
    "recommend",
    "reliable",
    "smooth",
    "solid",
    "stunning",
    "value",
    "worth",
}

NEGATIVE_WORDS = {
    "awful",
    "bad",
    "broke",
    "broken",
    "cheap",
    "complicated",
    "defective",
    "disappointing",
    "expensive",
    "fail",
    "failed",
    "fails",
    "flimsy",
    "frustrating",
    "garbage",
    "hate",
    "issue",
    "lag",
    "overpriced",
    "poor",
    "problem",
    "refund",
    "return",
    "slow",
    "terrible",
    "useless",
    "worst",
}


@dataclass(frozen=True, slots=True)
class SentimentBackend:
    model: Any
    backend_name: str
    model_name: str
    fallback_reason: str | None = None


class LexiconSentimentModel:
    def score(self, text: str | None) -> tuple[str, float]:
        return lexicon_score_text(text)


def _probe_transformers_import() -> tuple[bool, str | None]:
    if os.name != "nt":
        return True, None

    result = subprocess.run(
        [sys.executable, "-c", "from transformers import pipeline"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return True, None

    reason = result.stderr.strip() or result.stdout.strip() or f"subprocess exited with code {result.returncode}"
    return False, reason


def _normalize_model_label(label: Any) -> str:
    return str(label or "").strip().lower()


def _classify_signed_score(signed_score: float) -> str:
    if signed_score > 0.2:
        return "positive"
    if signed_score < -0.2:
        return "negative"
    return "neutral"


def _clamp_confidence(confidence: Any) -> float:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


def signed_score_from_transformer_output(result: Any) -> float:
    if isinstance(result, list) and result:
        top_result = max(
            (item for item in result if isinstance(item, dict)),
            key=lambda item: float(item.get("score", 0.0)),
            default={},
        )
        result = top_result

    if not isinstance(result, dict):
        return 0.0

    label = _normalize_model_label(result.get("label"))
    confidence = _clamp_confidence(result.get("score"))
    magnitude = (2.0 * confidence) - 1.0

    if label in POSITIVE_LABELS or "positive" in label:
        return round(magnitude, 4)
    if label in NEGATIVE_LABELS or "negative" in label:
        return round(-magnitude, 4)
    if label in NEUTRAL_LABELS or "neutral" in label:
        return 0.0
    return 0.0


def map_signed_score_to_label(signed_score: float) -> str:
    return _classify_signed_score(float(signed_score))


def lexicon_score_text(text: str | None) -> tuple[str, float]:
    tokens = TOKEN_PATTERN.findall((text or "").lower())
    if not tokens:
        return ("neutral", 0.0)

    positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return ("neutral", 0.0)

    signed_score = round((positive_hits - negative_hits) / total_hits, 4)
    return (_classify_signed_score(signed_score), signed_score)


def load_sentiment_backend(model_name: str | None = None) -> SentimentBackend:
    selected_model = (model_name or os.getenv("SENTIMENT_MODEL", DEFAULT_SENTIMENT_MODEL)).strip()
    selected_model = selected_model or DEFAULT_SENTIMENT_MODEL

    try:
        available, probe_reason = _probe_transformers_import()
        if not available:
            raise RuntimeError(f"transformers import probe failed: {probe_reason}")
        transformers = importlib.import_module("transformers")
        pipeline = getattr(transformers, "pipeline")
        model = pipeline(
            "sentiment-analysis",
            model=selected_model,
            tokenizer=selected_model,
        )
        return SentimentBackend(
            model=model,
            backend_name="transformers",
            model_name=selected_model,
        )
    except Exception as exc:
        return SentimentBackend(
            model=LexiconSentimentModel(),
            backend_name="lexicon",
            model_name="lexicon",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def score_texts(
    backend: SentimentBackend,
    texts: Sequence[str | None],
    *,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    normalized_texts = list(texts)
    results: list[tuple[str, float]] = [("neutral", 0.0) for _ in normalized_texts]

    non_empty_indices = [
        index for index, text in enumerate(normalized_texts)
        if text is not None and str(text).strip()
    ]
    if not non_empty_indices:
        return results

    if backend.backend_name == "transformers":
        batch_texts = [str(normalized_texts[index]) for index in non_empty_indices]
        predictions = backend.model(batch_texts, truncation=True, batch_size=batch_size)
        for index, prediction in zip(non_empty_indices, predictions):
            signed_score = signed_score_from_transformer_output(prediction)
            results[index] = (
                _classify_signed_score(signed_score),
                float(round(signed_score, 4)),
            )
        return results

    for index in non_empty_indices:
        results[index] = lexicon_score_text(str(normalized_texts[index]))
    return results
