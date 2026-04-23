from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import requests

from src.ml.sentiment_backend import lexicon_score_text


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

CANONICAL_ASPECT_KEYWORD_MAP: dict[str, tuple[str, ...]] = {
    "battery": (
        "battery",
        "battery life",
        "charge",
        "charging",
        "power",
        "lasts",
    ),
    "camera": (
        "camera",
        "photo",
        "picture",
        "photos",
        "pictures",
        "lens",
        "zoom",
    ),
    "sound quality": (
        "sound",
        "sound quality",
        "audio",
        "bass",
        "treble",
        "speaker",
        "noise cancellation",
        "anc",
    ),
    "build quality": (
        "build",
        "build quality",
        "construction",
        "material",
        "materials",
        "plastic",
        "metal",
        "sturdy",
        "flimsy",
    ),
    "comfort": (
        "comfort",
        "comfortable",
        "fit",
        "fits",
        "ear",
        "ergonomic",
        "lightweight",
        "heavy",
    ),
    "screen": (
        "screen",
        "display",
        "brightness",
        "resolution",
        "oled",
        "lcd",
        "refresh rate",
    ),
    "performance": (
        "performance",
        "speed",
        "fast",
        "slow",
        "lag",
        "laggy",
        "smooth",
        "processor",
        "cpu",
        "gpu",
    ),
    "price": (
        "price",
        "cost",
        "expensive",
        "cheap",
        "overpriced",
        "value",
        "worth",
        "affordable",
        "deal",
    ),
    "design": (
        "design",
        "look",
        "looks",
        "style",
        "aesthetic",
        "sleek",
        "beautiful",
        "ugly",
        "color",
    ),
    "durability": (
        "durable",
        "durability",
        "broke",
        "broken",
        "cracked",
        "lasted",
        "lasting",
        "wear",
    ),
    "customer service": (
        "customer service",
        "support",
        "warranty",
        "return",
        "refund",
        "replacement",
        "service team",
    ),
    "ease of use": (
        "easy",
        "simple",
        "intuitive",
        "complicated",
        "setup",
        "set up",
        "install",
        "installation",
        "user friendly",
    ),
}
ASPECT_KEYWORDS = CANONICAL_ASPECT_KEYWORD_MAP


def _normalize_phrase(value: str | None) -> str:
    lowered = str(value or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(lowered.split())


def canonicalize_aspect(value: str | None) -> str | None:
    normalized = _normalize_phrase(value)
    if not normalized:
        return None

    for canonical_label, keywords in CANONICAL_ASPECT_KEYWORD_MAP.items():
        candidate_phrases = (canonical_label, *keywords)
        if normalized == canonical_label:
            return canonical_label
        if normalized in candidate_phrases:
            return canonical_label
        if any(keyword in normalized for keyword in candidate_phrases):
            return canonical_label
    return None


def _trim_quote(sentence: str, *, max_chars: int = 240) -> str:
    normalized = " ".join(str(sentence or "").strip().split())
    return normalized[:max_chars]


def _normalize_aspect_records(aspects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    by_aspect: dict[str, int] = {}

    for aspect in aspects:
        canonical_label = canonicalize_aspect(aspect.get("aspect"))
        if canonical_label is None:
            continue

        quote = _trim_quote(str(aspect.get("quote", "")))
        try:
            sentiment = round(float(aspect.get("sentiment", 0.0)), 4)
        except (TypeError, ValueError):
            sentiment = 0.0

        normalized_item = {
            "aspect": canonical_label,
            "sentiment": sentiment,
            "quote": quote,
        }

        if canonical_label in by_aspect:
            existing_index = by_aspect[canonical_label]
            existing = normalized[existing_index]
            should_replace = abs(sentiment) > abs(float(existing.get("sentiment", 0.0)))
            if not existing.get("quote") and quote:
                should_replace = True
            if should_replace:
                normalized[existing_index] = normalized_item
            continue

        by_aspect[canonical_label] = len(normalized)
        normalized.append(normalized_item)

    return normalized


def extract_aspects_heuristic(text: str | None) -> list[dict[str, Any]]:
    source_text = str(text or "").strip()
    if not source_text:
        return []

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(source_text)
        if sentence and sentence.strip()
    ]
    extracted: list[dict[str, Any]] = []

    for sentence in sentences:
        normalized_sentence = _normalize_phrase(sentence)
        if len(normalized_sentence) < 4:
            continue

        _, sentence_score = lexicon_score_text(sentence)
        matched_labels: list[str] = []
        for canonical_label, keywords in CANONICAL_ASPECT_KEYWORD_MAP.items():
            phrases = (canonical_label, *keywords)
            if any(_normalize_phrase(keyword) in normalized_sentence for keyword in phrases):
                matched_labels.append(canonical_label)

        for canonical_label in matched_labels:
            extracted.append(
                {
                    "aspect": canonical_label,
                    "sentiment": sentence_score,
                    "quote": _trim_quote(sentence),
                }
            )

    return _normalize_aspect_records(extracted)


@lru_cache(maxsize=8)
def probe_ollama_host(
    ollama_host: str | None = None,
    timeout_seconds: int | None = None,
) -> bool:
    host = str(ollama_host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).strip().rstrip("/")
    if not host:
        return False

    timeout = int(timeout_seconds or os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _strip_markdown_fences(value: str) -> str:
    stripped = str(value or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_ollama_aspects(response_text: str) -> list[dict[str, Any]] | None:
    cleaned = _strip_markdown_fences(response_text)
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        candidate_items = payload.get("aspects", [])
    elif isinstance(payload, list):
        candidate_items = payload
    else:
        return None

    if not isinstance(candidate_items, list):
        return None

    return _normalize_aspect_records(
        [item for item in candidate_items if isinstance(item, dict)]
    )


def extract_aspects_with_ollama(
    text: str | None,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]] | None:
    source_text = str(text or "").strip()
    if not source_text:
        return []

    host = str(ollama_host or os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).strip().rstrip("/")
    model = str(ollama_model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip()
    timeout = int(timeout_seconds or os.getenv("OLLAMA_TIMEOUT_SECONDS", DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    allowed_aspects = ", ".join(CANONICAL_ASPECT_KEYWORD_MAP)
    prompt = (
        "Extract product aspects from the review below. "
        "Use only these canonical aspect labels when relevant: "
        f"{allowed_aspects}. "
        "Return ONLY JSON as an array of objects with keys aspect, sentiment, and quote. "
        "sentiment must be a number between -1 and 1.\n\n"
        f"Review: {source_text[:1500]}"
    )

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    parsed = _parse_ollama_aspects(payload.get("response", ""))
    return parsed


def serialize_aspects(aspects: list[dict[str, Any]] | None) -> tuple[str, int, str]:
    normalized = _normalize_aspect_records(list(aspects or []))
    labels = [item["aspect"] for item in normalized]
    aspect_labels_csv = ", ".join(labels)
    aspect_details_json = json.dumps(normalized, ensure_ascii=True)
    return (aspect_labels_csv, len(labels), aspect_details_json)
