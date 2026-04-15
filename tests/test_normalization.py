"""
ReviewPulse AI — Unit Tests for Schema Normalization
=====================================================
These tests verify the correctness of our normalization logic.
The professor asked: "What tests did you write to verify correctness?"
Here they are.

Run: pytest tests/test_normalization.py -v
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from poc.normalize_schema import (
    normalize_amazon,
    normalize_yelp,
    normalize_reddit,
    normalize_youtube,
    create_unified_record,
)


class TestAmazonNormalization:
    """Tests for Amazon Reviews 2023 schema normalization."""

    def test_rating_5_star_maps_to_1(self):
        """Amazon rating 5.0 should normalize to 1.0"""
        raw = {"rating": 5.0, "text": "Great", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": True}
        result = normalize_amazon(raw)
        assert result["rating_normalized"] == 1.0

    def test_rating_1_star_maps_to_0(self):
        """Amazon rating 1.0 should normalize to 0.0"""
        raw = {"rating": 1.0, "text": "Bad", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": False}
        result = normalize_amazon(raw)
        assert result["rating_normalized"] == 0.0

    def test_rating_3_star_maps_to_half(self):
        """Amazon rating 3.0 should normalize to 0.5"""
        raw = {"rating": 3.0, "text": "Ok", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": True}
        result = normalize_amazon(raw)
        assert result["rating_normalized"] == 0.5

    def test_timestamp_milliseconds_to_iso(self):
        """Amazon timestamps are epoch MS — should convert to ISO 8601"""
        raw = {"rating": 5.0, "text": "Good", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1588687728923, "helpful_vote": 0, "verified_purchase": True}
        result = normalize_amazon(raw)
        assert result["review_date"] is not None
        assert result["review_date"].startswith("2020-05-05")

    def test_null_text_does_not_crash(self):
        """12% of Amazon reviews have null text — should handle gracefully"""
        raw = {"rating": 5.0, "text": None, "title": None, "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": True}
        result = normalize_amazon(raw)
        assert result["review_text"] == ""
        assert result["text_length_words"] == 0

    def test_source_is_amazon(self):
        """Source field must be 'amazon'"""
        raw = {"rating": 5.0, "text": "Good", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": True}
        result = normalize_amazon(raw)
        assert result["source"] == "amazon"

    def test_verified_purchase_preserved(self):
        """verified_purchase boolean should pass through"""
        raw = {"rating": 5.0, "text": "Good", "asin": "B001", "parent_asin": "B001",
               "user_id": "U001", "timestamp": 1680000000000, "helpful_vote": 0, "verified_purchase": True}
        assert normalize_amazon(raw)["verified_purchase"] == True
        raw["verified_purchase"] = False
        assert normalize_amazon(raw)["verified_purchase"] == False


class TestYelpNormalization:
    """Tests for Yelp Open Dataset normalization."""

    def test_integer_stars_to_normalized(self):
        """Yelp uses integer 1-5 (not float like Amazon). Same formula."""
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 5, "text": "Great", "date": "2024-01-15"}
        result = normalize_yelp(raw)
        assert result["rating_normalized"] == 1.0

    def test_stars_1_maps_to_0(self):
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 1, "text": "Bad", "date": "2024-01-15"}
        result = normalize_yelp(raw)
        assert result["rating_normalized"] == 0.0

    def test_date_string_to_iso(self):
        """Yelp dates are 'YYYY-MM-DD' strings, not epoch timestamps"""
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 3, "text": "Ok", "date": "2024-06-15"}
        result = normalize_yelp(raw)
        assert result["review_date"] == "2024-06-15T00:00:00"

    def test_no_verified_purchase(self):
        """Yelp has no verified_purchase concept — must be None, not False"""
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 3, "text": "Ok", "date": "2024-01-15"}
        result = normalize_yelp(raw)
        assert result["verified_purchase"] is None

    def test_no_helpful_votes(self):
        """Yelp has no helpful_votes — must be None, not 0"""
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 3, "text": "Ok", "date": "2024-01-15"}
        result = normalize_yelp(raw)
        assert result["helpful_votes"] is None

    def test_source_is_yelp(self):
        raw = {"review_id": "y001", "user_id": "u001", "business_id": "b001",
               "stars": 3, "text": "Ok", "date": "2024-01-15"}
        assert normalize_yelp(raw)["source"] == "yelp"


class TestRedditNormalization:
    """Tests for Reddit API normalization."""

    def test_rating_is_null(self):
        """CRITICAL: Reddit has NO star rating. Must be None, not 0 or converted."""
        raw = {"source_id": "abc123", "title": "Review", "text": "Good product",
               "score": 500, "author": "user1", "created_utc": 1680000000,
               "subreddit": "headphones", "url": "/r/headphones/abc123"}
        result = normalize_reddit(raw)
        assert result["rating_normalized"] is None

    def test_score_not_converted_to_rating(self):
        """Post score (upvotes) must NOT be converted to a rating."""
        raw = {"source_id": "abc123", "title": "Review", "text": "Good",
               "score": 1000, "author": "user1", "created_utc": 1680000000,
               "subreddit": "headphones", "url": "/r/headphones/abc123"}
        result = normalize_reddit(raw)
        # Score goes to helpful_votes, NOT rating
        assert result["rating_normalized"] is None
        assert result["helpful_votes"] == 1000

    def test_timestamp_seconds_not_ms(self):
        """Reddit uses epoch SECONDS, Amazon uses MILLISECONDS. Don't mix them up."""
        raw = {"source_id": "abc123", "title": "Review", "text": "Good",
               "score": 10, "author": "user1", "created_utc": 1680000000,
               "subreddit": "headphones", "url": "/r/headphones/abc123"}
        result = normalize_reddit(raw)
        # Should be 2023, not 1970 (which would happen if treated as ms)
        assert result["review_date"] is not None
        assert result["review_date"].startswith("2023")

    def test_verified_purchase_is_null(self):
        """Reddit has no verified_purchase concept"""
        raw = {"source_id": "abc123", "title": "Review", "text": "Good",
               "score": 10, "author": "user1", "created_utc": 1680000000,
               "subreddit": "headphones", "url": "/r/headphones/abc123"}
        result = normalize_reddit(raw)
        assert result["verified_purchase"] is None

    def test_source_is_reddit(self):
        raw = {"source_id": "abc123", "title": "Review", "text": "Good",
               "score": 10, "author": "user1", "created_utc": 1680000000,
               "subreddit": "headphones", "url": "/r/headphones/abc123"}
        assert normalize_reddit(raw)["source"] == "reddit"


class TestYouTubeNormalization:
    """Tests for YouTube transcript normalization."""

    def test_rating_is_null(self):
        """YouTube transcripts have NO rating system"""
        raw = {"source_id": "vid001", "text": "Great headphones review",
               "created_utc": 1680000000, "url": "https://youtube.com/watch?v=vid001",
               "channel": "TechReviewer"}
        result = normalize_youtube(raw)
        assert result["rating_normalized"] is None

    def test_long_transcript_preserved(self):
        """YouTube transcripts are much longer than typical reviews"""
        long_text = "word " * 1000  # 1000 words
        raw = {"source_id": "vid001", "text": long_text,
               "created_utc": 1680000000, "url": "https://youtube.com/watch?v=vid001",
               "channel": "TechReviewer"}
        result = normalize_youtube(raw)
        assert result["text_length_words"] == 1000

    def test_source_is_youtube(self):
        raw = {"source_id": "vid001", "text": "Review",
               "created_utc": 1680000000, "url": "https://youtube.com/watch?v=vid001",
               "channel": "TechReviewer"}
        assert normalize_youtube(raw)["source"] == "youtube"


class TestUnifiedSchema:
    """Tests that verify the unified schema contract."""

    def test_all_required_fields_present(self):
        """Every normalized record must have all required fields"""
        record = create_unified_record(
            review_id="test_001", product_name="TestProduct",
            product_category="electronics", source="test",
            rating_normalized=0.75, review_text="Good product",
            review_date="2024-01-01T00:00:00", reviewer_id="user1",
            verified_purchase=True, helpful_votes=5,
            source_url="https://example.com",
        )
        required_fields = [
            "review_id", "product_name", "product_category", "source",
            "rating_normalized", "review_text", "review_date", "reviewer_id",
            "verified_purchase", "helpful_votes", "source_url", "text_length_words",
        ]
        for field in required_fields:
            assert field in record, f"Missing field: {field}"

    def test_rating_normalized_range(self):
        """rating_normalized must be 0-1 or None"""
        for rating in [0.0, 0.25, 0.5, 0.75, 1.0]:
            record = create_unified_record(
                review_id="t", product_name="p", product_category="c",
                source="s", rating_normalized=rating, review_text="text",
                review_date=None, reviewer_id="u", verified_purchase=None,
                helpful_votes=None, source_url="",
            )
            assert 0.0 <= record["rating_normalized"] <= 1.0

    def test_null_rating_allowed(self):
        """None rating must be allowed (for Reddit, YouTube)"""
        record = create_unified_record(
            review_id="t", product_name="p", product_category="c",
            source="reddit", rating_normalized=None, review_text="text",
            review_date=None, reviewer_id="u", verified_purchase=None,
            helpful_votes=None, source_url="",
        )
        assert record["rating_normalized"] is None


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_classes = [
        TestAmazonNormalization, TestYelpNormalization,
        TestRedditNormalization, TestYouTubeNormalization,
        TestUnifiedSchema,
    ]
    passed = 0
    failed = 0
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {cls.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {cls.__name__}.{method_name} — {e}")
                    failed += 1
    
    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
