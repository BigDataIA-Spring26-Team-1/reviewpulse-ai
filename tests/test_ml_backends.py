from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.aspect_extraction import (
    canonicalize_aspect,
    extract_aspects_heuristic,
    probe_ollama_host,
    serialize_aspects,
)
from src.ml.sentiment_backend import SentimentBackend, load_sentiment_backend, score_texts


def test_extract_aspects_heuristic_returns_canonical_labels_with_sentiment():
    aspects = extract_aspects_heuristic(
        "Battery life is amazing. Support was terrible. The design looks sleek."
    )

    assert [item["aspect"] for item in aspects] == [
        "battery",
        "customer service",
        "design",
    ]
    assert aspects[0]["sentiment"] > 0
    assert aspects[1]["sentiment"] < 0


def test_serialize_aspects_canonicalizes_and_deduplicates_labels():
    aspect_labels, aspect_count, aspect_details_json = serialize_aspects(
        [
            {"aspect": "battery life", "sentiment": 0.4, "quote": "Battery life is great."},
            {"aspect": "support", "sentiment": -0.6, "quote": "Support was disappointing."},
            {"aspect": "battery", "sentiment": 0.8, "quote": "Battery is incredible."},
        ]
    )

    details = json.loads(aspect_details_json)

    assert canonicalize_aspect("battery life") == "battery"
    assert canonicalize_aspect("support") == "customer service"
    assert aspect_labels == "battery, customer service"
    assert aspect_count == 2
    assert details[0]["aspect"] == "battery"
    assert details[0]["sentiment"] == pytest.approx(0.8)


def test_load_sentiment_backend_falls_back_to_lexicon(monkeypatch: pytest.MonkeyPatch):
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "transformers":
            raise ImportError("transformers unavailable")
        return original_import_module(name, package)

    monkeypatch.setattr("src.ml.sentiment_backend.importlib.import_module", fake_import_module)

    backend = load_sentiment_backend("distilbert-base-uncased-finetuned-sst-2-english")

    assert backend.backend_name == "lexicon"
    assert backend.model_name == "lexicon"
    assert backend.fallback_reason is not None


def test_score_texts_maps_transformer_outputs_to_signed_scores():
    class FakePipeline:
        def __call__(self, texts, truncation: bool = True, batch_size: int = 32):
            assert truncation is True
            assert batch_size == 32
            assert list(texts) == ["great", "mixed", "bad"]
            return [
                {"label": "POSITIVE", "score": 0.8},
                {"label": "NEGATIVE", "score": 0.55},
                {"label": "NEGATIVE", "score": 0.9},
            ]

    backend = SentimentBackend(
        model=FakePipeline(),
        backend_name="transformers",
        model_name="fake-transformer",
    )

    scored = score_texts(backend, ["great", "mixed", "bad"])

    assert scored[0][0] == "positive"
    assert scored[0][1] == pytest.approx(0.6)
    assert scored[1][0] == "neutral"
    assert scored[1][1] == pytest.approx(-0.1)
    assert scored[2][0] == "negative"
    assert scored[2][1] == pytest.approx(-0.8)


def test_probe_ollama_host_returns_false_when_host_is_unreachable(monkeypatch: pytest.MonkeyPatch):
    probe_ollama_host.cache_clear()

    def fake_get(*_args, **_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("src.ml.aspect_extraction.requests.get", fake_get)

    assert probe_ollama_host("http://localhost:11434", 1) is False
