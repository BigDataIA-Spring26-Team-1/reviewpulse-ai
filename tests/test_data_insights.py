"""Unit tests for the integrated data insights layer."""
 
from __future__ import annotations
 
from pathlib import Path
 
import pytest
 
from src.insights.data_profile import build_dataset_profile
from src.insights.normalization_explainer import build_normalization_explanations
from src.insights.quality_metrics import build_quality_metrics
from src.insights.source_comparison import build_source_comparison
 
 
SAMPLE_ROWS = [
    {
        "review_id": "amazon_1",
        "product_name": "Sony WH-1000XM5",
        "product_category": "electronics",
        "source": "amazon",
        "rating_normalized": 1.0,
        "review_text": "Great battery life",
        "review_date": "2024-01-01T00:00:00+00:00",
        "reviewer_id": "u1",
        "verified_purchase": True,
        "helpful_votes": 5,
        "source_url": "https://amazon.com/dp/B001",
        "text_length_words": 3,
    },
    {
        "review_id": "yelp_1",
        "product_name": "North End Cafe",
        "product_category": "Coffee & Tea, Cafes",
        "source": "yelp",
        "rating_normalized": 0.75,
        "review_text": "Great battery life",
        "review_date": "2024-01-02T00:00:00",
        "reviewer_id": "u2",
        "verified_purchase": None,
        "helpful_votes": None,
        "source_url": "https://yelp.com/biz/b001",
        "text_length_words": 3,
    },
    {
        "review_id": "youtube_1",
        "product_name": "Sony WH-1000XM5",
        "product_category": "audio",
        "source": "youtube",
        "rating_normalized": None,
        "review_text": "",
        "review_date": "bad-date",
        "reviewer_id": "channel-1",
        "verified_purchase": None,
        "helpful_votes": None,
        "source_url": "",
        "text_length_words": 0,
    },
]
 
 
def test_dataset_profile_summarizes_records():
    profile = build_dataset_profile(
        SAMPLE_ROWS,
        dataset_path="data/normalized_reviews_parquet",
    )
 
    assert profile["record_count"] == 3
    assert profile["source_count"] == 3
    assert profile["distinct_products"] == 2
    assert profile["distinct_reviewers"] == 3
    assert profile["average_review_length_words"] == 2.0
    assert profile["rating_normalized"]["mean"] == pytest.approx(0.875)
    assert profile["null_counts"]["rating_normalized"] == 1
    assert profile["date_range"]["min"] == "2024-01-01T00:00:00+00:00"
 
 
def test_source_comparison_reports_per_source_metrics():
    comparison = build_source_comparison(SAMPLE_ROWS)
    by_source = {row["source"]: row for row in comparison["sources"]}
 
    assert by_source["amazon"]["average_review_length_words"] == 3.0
    assert by_source["amazon"]["average_rating_normalized"] == 1.0
    assert by_source["youtube"]["average_rating_normalized"] is None
    assert by_source["youtube"]["missing_rating_ratio"] == 1.0
    assert by_source["youtube"]["missing_review_text_ratio"] == 1.0
 
 
def test_quality_metrics_detect_exact_duplicate_text_and_invalid_dates():
    quality = build_quality_metrics(SAMPLE_ROWS)
    by_source = {row["source"]: row for row in quality["per_source"]}
 
    assert quality["duplicate_ratio"] == pytest.approx(0.3333, rel=1e-3)
    assert quality["empty_text_ratio"] == pytest.approx(0.3333, rel=1e-3)
    assert quality["invalid_date_count"] == 1
    assert quality["missing_rating_count"] == 1
    assert "MinHash/LSH" in quality["duplicate_method"]
    assert by_source["youtube"]["invalid_date_count"] == 1
 
 
def test_normalization_explainer_returns_amazon_sample(monkeypatch: pytest.MonkeyPatch):
    sample_row = {
        "rating": 5.0,
        "asin": "B001",
        "parent_asin": "B001",
        "user_id": "U001",
        "timestamp": 1680000000000,
        "text": "Great headphones",
    }
 
    monkeypatch.setattr(
        "src.insights.normalization_explainer._load_sample_raw_row",
        lambda source, sample_index=0: (sample_row, Path("data/amazon.jsonl")),
    )
 
    payload = build_normalization_explanations(source="amazon")
    explanation = payload["explanations"][0]
 
    assert explanation["source"] == "amazon"
    assert explanation["formula"] == "(rating - 1.0) / 4.0"
    assert explanation["sample_raw_rating_value"] == 5.0
    assert explanation["sample_normalized_rating"] == 1.0
    assert explanation["sample_identifiers"]["product_name"] == "B001"
 
 
def test_normalization_explainer_keeps_youtube_rating_null(monkeypatch: pytest.MonkeyPatch):
    sample_row = {
        "source_id": "vid001",
        "title": "Sony XM5 Review",
        "text": "Transcript text",
        "created_utc": 1680000000,
        "channel": "TechReviewer",
        "url": "https://youtube.com/watch?v=vid001",
    }
 
    monkeypatch.setattr(
        "src.insights.normalization_explainer._load_sample_raw_row",
        lambda source, sample_index=0: (sample_row, Path("data/youtube_reviews.jsonl")),
    )
 
    payload = build_normalization_explanations(source="youtube")
    explanation = payload["explanations"][0]
 
    assert explanation["formula"] == "rating_normalized = null"
    assert explanation["sample_raw_rating_value"] is None
    assert explanation["sample_normalized_rating"] is None
    assert explanation["sample_raw_date_value"] == 1680000000
 