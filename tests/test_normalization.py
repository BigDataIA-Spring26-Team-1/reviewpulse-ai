"""Unit tests for proposal-aligned schema normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.core import (
    UNIFIED_REVIEW_FIELDS,
    create_unified_record,
    normalize_amazon,
    normalize_ebay,
    normalize_ifixit,
    normalize_reddit,
    normalize_yelp,
    normalize_youtube,
)


class TestAmazonNormalization:
    def test_rating_5_star_maps_to_1(self):
        result = normalize_amazon(
            {
                "rating": 5.0,
                "asin": "B001",
                "parent_asin": "B001",
                "user_id": "U001",
                "timestamp": 1680000000000,
                "helpful_vote": 0,
                "verified_purchase": True,
                "text": "Great",
            }
        )
        assert result["rating_normalized"] == 1.0

    def test_rating_3_star_maps_to_half(self):
        result = normalize_amazon(
            {
                "rating": 3.0,
                "asin": "B001",
                "parent_asin": "B001",
                "user_id": "U001",
                "timestamp": 1680000000000,
                "text": "Ok",
            }
        )
        assert result["rating_normalized"] == 0.5

    def test_timestamp_milliseconds_to_iso(self):
        result = normalize_amazon(
            {
                "rating": 5.0,
                "asin": "B001",
                "parent_asin": "B001",
                "user_id": "U001",
                "timestamp": 1588687728923,
                "text": "Good",
            }
        )
        assert result["review_date"].startswith("2020-05-05")

    def test_null_text_does_not_crash(self):
        result = normalize_amazon(
            {
                "rating": 5.0,
                "asin": "B001",
                "parent_asin": "B001",
                "user_id": "U001",
                "timestamp": 1680000000000,
                "text": None,
                "title": None,
            }
        )
        assert result["review_text"] == ""
        assert result["text_length_words"] == 0


class TestYelpNormalization:
    def test_integer_stars_to_normalized(self):
        result = normalize_yelp(
            {
                "review_id": "y001",
                "user_id": "u001",
                "business_id": "b001",
                "stars": 5,
                "text": "Great",
                "date": "2024-01-15",
            }
        )
        assert result["rating_normalized"] == 1.0

    def test_date_string_to_iso(self):
        result = normalize_yelp(
            {
                "review_id": "y001",
                "user_id": "u001",
                "business_id": "b001",
                "stars": 3,
                "text": "Ok",
                "date": "2024-06-15",
            }
        )
        assert result["review_date"] == "2024-06-15T00:00:00"

    def test_business_lookup_enriches_name(self):
        result = normalize_yelp(
            {
                "review_id": "y001",
                "user_id": "u001",
                "business_id": "b001",
                "stars": 4,
                "text": "Great service",
                "date": "2024-06-15",
            },
            business_lookup={
                "b001": {
                    "business_id": "b001",
                    "name": "North End Cafe",
                    "categories": "Coffee & Tea, Cafes",
                }
            },
        )
        assert result["product_name"] == "North End Cafe"
        assert result["display_category"] == "Coffee & Tea"


class TestEbayNormalization:
    def test_seller_rating_percentage_maps_to_unit_interval(self):
        result = normalize_ebay(
            {
                "item_id": "123456789",
                "title": "Sony WH-1000XM5 Wireless Headphones",
                "seller_rating": 99.2,
                "feedback_text": "Great headphones, fast shipping.",
                "category": "Electronics",
                "listing_date": "2024-11-15T10:30:00Z",
            }
        )
        assert result["rating_normalized"] == pytest.approx(0.992)

    def test_item_url_defaults_from_item_id(self):
        result = normalize_ebay(
            {
                "item_id": "123456789",
                "title": "Sony WH-1000XM5 Wireless Headphones",
                "seller_rating": 99.2,
                "feedback_text": "Great headphones, fast shipping.",
                "category": "Electronics",
                "listing_date": "2024-11-15T10:30:00Z",
            }
        )
        assert result["source_url"].endswith("/123456789")


class TestIFixitNormalization:
    def test_score_8_maps_to_expected_value(self):
        result = normalize_ifixit(
            {
                "guide_id": "ifixit_142085",
                "title": "iPhone 15 Pro Teardown",
                "repairability_score": 8,
                "review_text": "Battery replacement was straightforward.",
                "device_category": "Smartphones",
                "device_name": "iPhone 15 Pro",
                "author": "ifixit_user_4821",
                "published_date": "2024-09-25T14:00:00Z",
            }
        )
        assert result["rating_normalized"] == pytest.approx(0.7778)

    def test_score_1_maps_to_zero(self):
        result = normalize_ifixit(
            {
                "guide_id": "ifixit_142085",
                "repairability_score": 1,
                "review_text": "Hard to repair.",
            }
        )
        assert result["rating_normalized"] == 0.0

    def test_score_10_maps_to_one(self):
        result = normalize_ifixit(
            {
                "guide_id": "ifixit_142085",
                "repairability_score": 10,
                "review_text": "Very easy to repair.",
            }
        )
        assert result["rating_normalized"] == 1.0


class TestRedditNormalization:
    def test_score_is_not_converted_to_rating(self):
        result = normalize_reddit(
            {
                "source_id": "abc123",
                "title": "Review",
                "text": "Good product",
                "score": 500,
                "author": "user1",
                "created_utc": 1680000000,
                "subreddit": "headphones",
                "url": "/r/headphones/abc123",
            }
        )
        assert result["rating_normalized"] is None
        assert result["helpful_votes"] == 500


class TestYoutubeNormalization:
    def test_rating_is_null(self):
        result = normalize_youtube(
            {
                "source_id": "vid001",
                "text": "Great headphones review",
                "created_utc": 1680000000,
                "url": "https://youtube.com/watch?v=vid001",
                "channel": "TechReviewer",
            }
        )
        assert result["rating_normalized"] is None

    def test_long_transcript_preserved(self):
        result = normalize_youtube(
            {
                "source_id": "vid001",
                "text": "word " * 1000,
                "created_utc": 1680000000,
                "url": "https://youtube.com/watch?v=vid001",
                "channel": "TechReviewer",
            }
        )
        assert result["text_length_words"] == 1000


class TestUnifiedSchema:
    def test_all_required_fields_present(self):
        record = create_unified_record(
            review_id="test_001",
            product_name="TestProduct",
            product_category="electronics",
            source="test",
            rating_normalized=0.75,
            review_text="Good product",
            review_date="2024-01-01T00:00:00",
            reviewer_id="user1",
            verified_purchase=True,
            helpful_votes=5,
            source_url="https://example.com",
            display_name="Test Product",
            display_category="Electronics",
            entity_type="product_review",
        )
        for field_name in UNIFIED_REVIEW_FIELDS:
            assert field_name in record

    def test_null_rating_allowed(self):
        record = create_unified_record(
            review_id="t",
            product_name="p",
            product_category="c",
            source="reddit",
            rating_normalized=None,
            review_text="text",
            review_date=None,
            reviewer_id="u",
            verified_purchase=None,
            helpful_votes=None,
            source_url="",
        )
        assert record["rating_normalized"] is None
