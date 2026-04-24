from __future__ import annotations
 
from pathlib import Path
 
import pyarrow as pa
import pyarrow.parquet as pq
 
import src.insights.app_analytics as app_analytics
from src.insights.app_analytics import (
    ReviewFilter,
    build_aspect_intelligence,
    build_sentiment_analytics,
    compare_products,
    explore_reviews,
    list_filter_options,
)
 
 
def _write_reviews(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "review_id": ["r1", "r2", "r3"],
            "product_name": ["Phone A", "Phone A", "Headphones B"],
            "product_category": ["electronics", "electronics", "audio"],
            "source": ["amazon", "yelp", "amazon"],
            "rating_normalized": [0.9, 0.2, 0.8],
            "review_text": [
                "Battery life is excellent and customer support helped quickly.",
                "Battery failed and customer service was slow.",
                "Sound quality is strong and comfort is excellent.",
            ],
            "review_date": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-15T00:00:00+00:00",
                "2026-02-01T00:00:00+00:00",
            ],
            "reviewer_id": ["u1", "u2", "u3"],
            "verified_purchase": [True, None, True],
            "helpful_votes": [5, 1, 8],
            "source_url": ["https://example.test/r1", "", "https://example.test/r3"],
            "display_name": ["Phone A Review", "Phone A Review", "Headphones B Review"],
            "display_category": ["Electronics", "Electronics", "Audio"],
            "entity_type": ["product_review", "business_review", "product_review"],
            "aspect_labels": ["battery, customer service", "battery, customer service", "sound quality, comfort"],
            "aspect_count": [2, 2, 2],
            "sentiment_label": ["positive", "negative", "positive"],
            "sentiment_score": [0.95, 0.12, 0.88],
        }
    )
    pq.write_table(table, path / "part-00000.parquet", compression="snappy")
 
 
def test_review_explorer_filters_rows(tmp_path: Path):
    dataset_path = tmp_path / "reviews"
    _write_reviews(dataset_path)
 
    payload = explore_reviews(
        dataset_path,
        ReviewFilter(query="battery", source="amazon"),
        limit=10,
        offset=0,
    )
 
    assert payload["matched_count"] == 1
    assert payload["rows"][0]["review_id"] == "r1"
    assert payload["rows"][0]["sentiment_label"] == "positive"
    assert payload["rows"][0]["product_label"] == "Phone A"
 
 
def test_filter_options_include_products_sources_and_date_range(tmp_path: Path):
    dataset_path = tmp_path / "reviews"
    _write_reviews(dataset_path)
 
    payload = list_filter_options(dataset_path)
 
    assert payload["products"][0]["product_name"] == "Phone A"
    assert payload["products"][0]["product_label"] == "Phone A"
    assert {row["source"] for row in payload["sources"]} == {"amazon", "yelp"}
    assert payload["date_range"]["min"].startswith("2026-01-01")
 
 
def test_sentiment_analytics_builds_chart_payloads(tmp_path: Path):
    dataset_path = tmp_path / "reviews"
    _write_reviews(dataset_path)
 
    payload = build_sentiment_analytics(dataset_path, ReviewFilter())
 
    sentiments = {row["sentiment_label"]: row["count"] for row in payload["sentiment_distribution"]}
    assert sentiments["positive"] == 2
    assert sentiments["negative"] == 1
    assert payload["monthly_trend"]
    assert payload["top_aspects"][0]["aspect"] == "battery"
 
 
def test_product_comparison_aggregates_requested_products(tmp_path: Path):
    dataset_path = tmp_path / "reviews"
    _write_reviews(dataset_path)
 
    payload = compare_products(
        dataset_path,
        ["Phone A", "Headphones B"],
        ReviewFilter(),
    )
 
    by_product = {row["product_name"]: row for row in payload["products"]}
    assert by_product["Phone A"]["review_count"] == 2
    assert by_product["Phone A"]["product_label"] == "Phone A"
    assert by_product["Phone A"]["average_rating_normalized"] == 0.55
    assert by_product["Headphones B"]["top_positive_aspects"][0]["aspect"] == "sound quality"
 
 
def test_aspect_intelligence_summarizes_positive_and_negative_aspects(tmp_path: Path):
    dataset_path = tmp_path / "reviews"
    _write_reviews(dataset_path)
 
    payload = build_aspect_intelligence(dataset_path, ReviewFilter())
 
    by_aspect = {row["aspect"]: row for row in payload["aspects"]}
    assert by_aspect["battery"]["positive_count"] == 1
    assert by_aspect["battery"]["negative_count"] == 1
    assert payload["top_negative_aspects"][0]["aspect"] in {"battery", "customer service"}
 
 
def test_amazon_machine_ids_get_friendly_product_labels(
    tmp_path: Path,
    monkeypatch,
):
    dataset_path = tmp_path / "reviews"
    dataset_path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "review_id": ["r1"],
            "product_name": ["B01G8JO5F2"],
            "product_category": ["electronics"],
            "source": ["amazon"],
            "rating_normalized": [0.9],
            "review_text": ["Great workout earbuds."],
            "review_date": ["2026-01-01T00:00:00+00:00"],
            "reviewer_id": ["u1"],
            "verified_purchase": [True],
            "helpful_votes": [3],
            "source_url": ["https://amazon.com/dp/B01G8JO5F2"],
            "display_name": ["Great for running"],
            "display_category": ["Electronics"],
            "entity_type": ["product_review"],
            "aspect_labels": ["comfort"],
            "aspect_count": [1],
            "sentiment_label": ["positive"],
            "sentiment_score": [0.82],
        }
    )
    pq.write_table(table, dataset_path / "part-00000.parquet", compression="snappy")
 
    monkeypatch.setattr(app_analytics, "resolve_amazon_product_titles", lambda product_names: {})
 
    options_payload = list_filter_options(dataset_path)
    explore_payload = explore_reviews(dataset_path, ReviewFilter(), limit=10, offset=0)
    compare_payload = compare_products(dataset_path, ["B01G8JO5F2"], ReviewFilter())
 
    assert options_payload["products"][0]["product_label"] == "Amazon Product (ASIN B01G8JO5F2)"
    assert explore_payload["rows"][0]["product_label"] == "Amazon Product (ASIN B01G8JO5F2)"
    assert compare_payload["products"][0]["product_label"] == "Amazon Product (ASIN B01G8JO5F2)"
 
 
def test_amazon_metadata_lookup_prefers_exact_product_title(
    tmp_path: Path,
    monkeypatch,
):
    dataset_path = tmp_path / "reviews"
    dataset_path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "review_id": ["r1"],
            "product_name": ["B075X8471B"],
            "product_category": ["electronics"],
            "source": ["amazon"],
            "rating_normalized": [0.9],
            "review_text": ["Solid streaming stick."],
            "review_date": ["2026-01-01T00:00:00+00:00"],
            "reviewer_id": ["u1"],
            "verified_purchase": [True],
            "helpful_votes": [4],
            "source_url": ["https://amazon.com/dp/B075X8471B"],
            "display_name": ["Worth buying"],
            "display_category": ["Electronics"],
            "entity_type": ["product_review"],
            "aspect_labels": ["performance"],
            "aspect_count": [1],
            "sentiment_label": ["positive"],
            "sentiment_score": [0.81],
        }
    )
    pq.write_table(table, dataset_path / "part-00000.parquet", compression="snappy")
 
    monkeypatch.setattr(
        app_analytics,
        "resolve_amazon_product_titles",
        lambda product_names: {
            product_name: "Fire TV Stick with Alexa Voice Remote"
            for product_name in product_names
        },
    )
 
    options_payload = list_filter_options(dataset_path)
    explore_payload = explore_reviews(dataset_path, ReviewFilter(), limit=10, offset=0)
    compare_payload = compare_products(dataset_path, ["B075X8471B"], ReviewFilter())
 
    assert options_payload["products"][0]["product_label"] == "Fire TV Stick with Alexa Voice Remote"
    assert options_payload["products"][0]["product_title"] == "Fire TV Stick with Alexa Voice Remote"
    assert explore_payload["rows"][0]["product_label"] == "Fire TV Stick with Alexa Voice Remote"
    assert explore_payload["rows"][0]["product_title"] == "Fire TV Stick with Alexa Voice Remote"
    assert compare_payload["products"][0]["product_label"] == "Fire TV Stick with Alexa Voice Remote"
    assert compare_payload["products"][0]["product_title"] == "Fire TV Stick with Alexa Voice Remote"
 
 