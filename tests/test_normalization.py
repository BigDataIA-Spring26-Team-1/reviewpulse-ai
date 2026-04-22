"""Unit tests for shared schema normalization."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyarrow.parquet as pq


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
    resolve_amazon_source_path,
    resolve_yelp_source_paths,
)
import src.spark.normalize_reviews_spark as spark_normalization


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

    def test_resolve_amazon_source_path_falls_back_to_latest_completed_run_dir(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "amazon_latest_run"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            run_dir = workspace / "amazon" / "runs" / "run_test"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "amazon_reviews_batch_000001_records_00000001.jsonl").write_text(
                '{"asin":"B001","user_id":"u1","rating":5.0}\n',
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"total_batches": 1, "total_records": 1}),
                encoding="utf-8",
            )
            (run_dir / "_checkpoint.json").write_text(
                json.dumps({"completed": True, "cumulative_records": 1}),
                encoding="utf-8",
            )

            resolved = resolve_amazon_source_path(workspace)

            assert resolved == run_dir.resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_resolve_amazon_source_path_prefers_completed_run_over_legacy_file(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "amazon_prefers_run"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            legacy_file = workspace / "amazon" / "amazon_reviews.jsonl"
            legacy_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_file.write_text('{"asin":"legacy"}\n', encoding="utf-8")

            run_dir = workspace / "amazon" / "runs" / "run_test"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "amazon_reviews_batch_000001_records_00000001.jsonl").write_text(
                '{"asin":"B001","user_id":"u1","rating":5.0}\n',
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"total_batches": 1, "total_records": 1}),
                encoding="utf-8",
            )
            (run_dir / "_checkpoint.json").write_text(
                json.dumps({"completed": True, "cumulative_records": 1}),
                encoding="utf-8",
            )

            resolved = resolve_amazon_source_path(workspace)

            assert resolved == run_dir.resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_resolve_amazon_source_path_falls_back_to_legacy_run_dir(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "amazon_legacy_run"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            run_dir = workspace / "amazon" / "runs" / "run_legacy"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "amazon_reviews_batch_00000_size_010000.jsonl").write_text(
                '{"asin":"B001","user_id":"u1","rating":5.0}\n',
                encoding="utf-8",
            )
            (run_dir / "amazon_reviews_batch_00001_size_010000.jsonl").write_text(
                '{"asin":"B002","user_id":"u2","rating":4.0}\n',
                encoding="utf-8",
            )

            resolved = resolve_amazon_source_path(workspace)

            assert resolved == run_dir.resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_spark_optional_column_defaults_missing_field_to_null_literal(self, monkeypatch: pytest.MonkeyPatch):
        operations: list[tuple[str, object]] = []

        class FakeExpression:
            def __init__(self, label: str) -> None:
                self.label = label

            def cast(self, cast_type: str):
                operations.append(("cast", cast_type))
                return f"{self.label}.cast({cast_type})"

        monkeypatch.setattr(
            spark_normalization,
            "col",
            lambda column_name: operations.append(("col", column_name)) or FakeExpression(f"col({column_name})"),
        )
        monkeypatch.setattr(
            spark_normalization,
            "lit",
            lambda value: operations.append(("lit", value)) or FakeExpression(f"lit({value})"),
        )

        expression = spark_normalization._optional_column(
            SimpleNamespace(columns=["asin", "title"]),
            "url",
            cast_type="string",
        )

        assert expression == "lit(None).cast(string)"
        assert operations == [
            ("lit", None),
            ("cast", "string"),
        ]

    def test_write_parquet_via_arrow_writes_partition_files(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "arrow_parquet_writer"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            output_path = workspace / "normalized_reviews_parquet"
            schema = SimpleNamespace(
                fields=[
                    SimpleNamespace(name="review_id", dataType=SimpleNamespace(typeName=lambda: "string")),
                    SimpleNamespace(name="rating_normalized", dataType=SimpleNamespace(typeName=lambda: "double")),
                    SimpleNamespace(name="verified_purchase", dataType=SimpleNamespace(typeName=lambda: "boolean")),
                ]
            )

            class FakeRow:
                def __init__(self, payload: dict[str, object]) -> None:
                    self._payload = payload

                def asDict(self, recursive: bool = False) -> dict[str, object]:
                    return dict(self._payload)

            class FakeDataFrame:
                def __init__(self) -> None:
                    self.schema = schema

                def toLocalIterator(self):
                    return iter(
                        [
                            FakeRow({"review_id": "r1", "rating_normalized": 1.0, "verified_purchase": True}),
                            FakeRow({"review_id": "r2", "rating_normalized": 0.5, "verified_purchase": None}),
                            FakeRow({"review_id": "r3", "rating_normalized": None, "verified_purchase": False}),
                        ]
                    )

            logger = logging.getLogger("reviewpulse.tests.spark.arrow")
            run_context = SimpleNamespace(stage="normalize_reviews_spark", run_id="run_test", dag_id=None, task_id=None)

            record_count = spark_normalization._write_parquet_via_arrow(
                FakeDataFrame(),
                output_path,
                logger=logger,
                run_context=run_context,
                batch_size=2,
            )

            files = sorted(output_path.glob("*.parquet"))
            assert record_count == 3
            assert [path.name for path in files] == ["part-00000.parquet", "part-00001.parquet"]
            rows = []
            for path in files:
                rows.extend(pq.read_table(path).to_pylist())
            assert rows[0]["review_id"] == "r1"
            assert rows[1]["verified_purchase"] is None
            assert rows[2]["verified_purchase"] is False
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


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

    def test_resolve_yelp_source_paths_prefers_current_pair_over_root_files(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "yelp_prefers_current"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            current_dir = workspace / "yelp" / "current"
            current_dir.mkdir(parents=True, exist_ok=True)
            (current_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"current"}\n',
                encoding="utf-8",
            )
            (current_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"current"}\n',
                encoding="utf-8",
            )

            root_dir = workspace / "yelp"
            (root_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"root"}\n',
                encoding="utf-8",
            )
            (root_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"root"}\n',
                encoding="utf-8",
            )

            review_path, business_path = resolve_yelp_source_paths(workspace)

            assert review_path == (current_dir / "yelp_academic_dataset_review.json").resolve()
            assert business_path == (current_dir / "yelp_academic_dataset_business.json").resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_resolve_yelp_source_paths_prefers_completed_run_over_legacy_files(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "yelp_prefers_run"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            root_dir = workspace / "yelp"
            root_dir.mkdir(parents=True, exist_ok=True)
            (root_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"root"}\n',
                encoding="utf-8",
            )
            (root_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"root"}\n',
                encoding="utf-8",
            )

            run_dir = workspace / "yelp" / "runs" / "run_test"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"run"}\n',
                encoding="utf-8",
            )
            (run_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"run"}\n',
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"total_files": 2, "total_records": 1}),
                encoding="utf-8",
            )

            review_path, business_path = resolve_yelp_source_paths(workspace)

            assert review_path == (run_dir / "yelp_academic_dataset_review.json").resolve()
            assert business_path == (run_dir / "yelp_academic_dataset_business.json").resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_resolve_yelp_source_paths_falls_back_to_root_pair(self):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "yelp_root_pair"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            root_dir = workspace / "yelp"
            root_dir.mkdir(parents=True, exist_ok=True)
            (root_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"root"}\n',
                encoding="utf-8",
            )
            (root_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"root"}\n',
                encoding="utf-8",
            )

            review_path, business_path = resolve_yelp_source_paths(workspace)

            assert review_path == (root_dir / "yelp_academic_dataset_review.json").resolve()
            assert business_path == (root_dir / "yelp_academic_dataset_business.json").resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_resolve_yelp_source_paths_falls_back_to_configured_dataset_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        workspace = Path(__file__).resolve().parent / "_tmp_normalization" / "yelp_configured_dataset"
        shutil.rmtree(workspace, ignore_errors=True)
        try:
            dataset_dir = workspace / "external_yelp_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "yelp_academic_dataset_review.json").write_text(
                '{"review_id":"configured"}\n',
                encoding="utf-8",
            )
            (dataset_dir / "yelp_academic_dataset_business.json").write_text(
                '{"business_id":"configured"}\n',
                encoding="utf-8",
            )

            monkeypatch.setattr(
                "src.normalization.core.get_settings",
                lambda: SimpleNamespace(yelp_dataset_path=dataset_dir.resolve()),
            )

            review_path, business_path = resolve_yelp_source_paths(workspace)

            assert review_path == (dataset_dir / "yelp_academic_dataset_review.json").resolve()
            assert business_path == (dataset_dir / "yelp_academic_dataset_business.json").resolve()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


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
