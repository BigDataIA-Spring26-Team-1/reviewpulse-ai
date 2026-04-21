from __future__ import annotations
 
import logging
import shutil
from dataclasses import replace
from io import StringIO
from pathlib import Path
 
import pytest
 
from src.common.run_context import build_run_context
from src.common.settings import Settings
from src.common.storage import S3PathResolver, S3StorageManager
from src.common.structured_logging import configure_structured_logging
from src.ingestion.amazon import build_dataset_config, run as run_amazon
from src.ingestion.common import count_lines
from src.ingestion.ebay import map_item_summary, run as run_ebay
from src.ingestion.ifixit import map_guide_payload
from src.ingestion.youtube import build_record, resolve_video_ids
from src.ingestion.yelp import BUSINESS_FILENAME, REVIEW_FILENAME, resolve_yelp_source_files, run as run_yelp
from tests.test_storage_foundation import FakeS3Client
 
 
def make_workspace(name: str) -> Path:
    workspace = Path(__file__).resolve().parent / "_tmp_source_ingestion" / name
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
 
 
def build_test_settings(workspace: Path) -> Settings:
    data_dir = workspace / "data"
    return Settings(
        project_root=workspace,
        app_name="ReviewPulse AI",
        app_env="test",
        log_level="INFO",
        data_dir=data_dir,
        raw_data_dir=data_dir / "raw",
        processed_data_dir=data_dir / "processed",
        normalized_jsonl_path=data_dir / "normalized_reviews.jsonl",
        normalized_parquet_path=data_dir / "normalized_reviews_parquet",
        sentiment_parquet_path=data_dir / "reviews_with_sentiment_parquet",
        chroma_path=data_dir / "chromadb_reviews",
        aws_region="us-east-1",
        s3_bucket_name="reviewpulse-bucket",
        s3_raw_prefix="raw",
        s3_processed_prefix="processed",
        spark_master="local[*]",
        spark_sql_session_timezone="UTC",
        huggingface_token="",
        amazon_dataset_name="McAuley-Lab/Amazon-Reviews-2023",
        amazon_category="Electronics",
        amazon_max_records=2,
        yelp_dataset_path=None,
        ebay_app_id="app-id",
        ebay_dev_id="dev-id",
        ebay_cert_id="cert-id",
        ebay_site_id="0",
        ebay_marketplace_id="EBAY_US",
        ebay_search_queries=("sony headphones",),
        ebay_max_items_per_query=2,
        ifixit_base_url="https://www.ifixit.com",
        ifixit_guide_ids=("12345",),
        youtube_api_key="",
        youtube_video_ids=("https://youtu.be/abcdefghijk",),
        youtube_transcript_languages=("en",),
    )
 
 
def build_storage_manager() -> S3StorageManager:
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    return S3StorageManager(resolver, FakeS3Client())
 
 
def build_logger() -> logging.Logger:
    stream = StringIO()
    configure_structured_logging("INFO", logger_name="reviewpulse.tests.ingestion", stream=stream)
    return logging.getLogger("reviewpulse.tests.ingestion")
 
 
def test_build_dataset_config_normalizes_amazon_category():
    assert build_dataset_config("Electronics") == "raw_review_Electronics"
    assert build_dataset_config("raw_review_Books") == "raw_review_Books"
 
 
def test_resolve_yelp_source_files_accepts_directory():
    workspace = make_workspace("resolve_yelp")
    try:
        dataset_dir = workspace / "yelp_source"
        dataset_dir.mkdir()
        (dataset_dir / REVIEW_FILENAME).write_text('{"review_id":"1"}\n', encoding="utf-8")
        (dataset_dir / BUSINESS_FILENAME).write_text('{"business_id":"b1"}\n', encoding="utf-8")
 
        review_path, business_path = resolve_yelp_source_files(dataset_dir)
 
        assert review_path.name == REVIEW_FILENAME
        assert business_path.name == BUSINESS_FILENAME
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
 
 
def test_run_yelp_copies_and_publishes_source_files():
    workspace = make_workspace("run_yelp")
    try:
        dataset_dir = workspace / "yelp_dataset"
        dataset_dir.mkdir()
        (dataset_dir / REVIEW_FILENAME).write_text('{"review_id":"1"}\n{"review_id":"2"}\n', encoding="utf-8")
        (dataset_dir / BUSINESS_FILENAME).write_text('{"business_id":"b1"}\n', encoding="utf-8")
 
        settings = replace(build_test_settings(workspace), yelp_dataset_path=dataset_dir)
        result = run_yelp(
            settings=settings,
            run_context=build_run_context(stage="ingest_yelp", source="yelp", run_id="run_test"),
            logger=build_logger(),
            storage_manager=build_storage_manager(),
        )
 
        local_review = workspace / "data" / "yelp" / REVIEW_FILENAME
        assert local_review.exists()
        assert count_lines(local_review) == 2
        assert result.record_count == 2
        assert result.file_count == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
 
 
def test_run_amazon_writes_and_publishes_records(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("run_amazon")
    try:
        settings = build_test_settings(workspace)
 
        monkeypatch.setattr(
            "src.ingestion.amazon.fetch_amazon_records",
            lambda _settings: [
                {"asin": "B001", "user_id": "u1", "rating": 5.0},
                {"asin": "B002", "user_id": "u2", "rating": 4.0},
            ],
        )
 
        result = run_amazon(
            settings=settings,
            run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
            logger=build_logger(),
            storage_manager=build_storage_manager(),
        )
 
        output_path = workspace / "data" / "amazon" / "amazon_reviews.jsonl"
        assert output_path.exists()
        assert count_lines(output_path) == 2
        assert result.record_count == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
 
 
def test_map_item_summary_keeps_real_ebay_fields():
    record = map_item_summary(
        {
            "itemId": "v1|123|0",
            "title": "Sony WH-1000XM5",
            "shortDescription": "Wireless noise-canceling headphones.",
            "categories": [{"categoryName": "Headphones"}],
            "itemCreationDate": "2026-04-19T12:00:00Z",
            "itemWebUrl": "https://www.ebay.com/itm/123",
            "seller": {"username": "seller1", "feedbackPercentage": "99.8", "feedbackScore": 4123},
        },
        "sony headphones",
    )
 
    assert record["item_id"] == "v1|123|0"
    assert record["seller_rating"] == "99.8"
    assert record["feedback_text"] == "Wireless noise-canceling headphones."
 
 
def test_run_ebay_requires_queries():
    workspace = make_workspace("run_ebay")
    try:
        settings = replace(build_test_settings(workspace), ebay_search_queries=tuple())
 
        with pytest.raises(RuntimeError, match="EBAY_SEARCH_QUERIES"):
            run_ebay(
                settings=settings,
                run_context=build_run_context(stage="ingest_ebay", source="ebay", run_id="run_test"),
                logger=build_logger(),
                storage_manager=build_storage_manager(),
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
 
 
def test_map_guide_payload_preserves_ifixit_fields():
    record = map_guide_payload(
        {
            "guideid": "12345",
            "title": "iPhone 15 Pro Teardown",
            "subject": "iPhone 15 Pro",
            "category": "Smartphones",
            "repairability_score": 8,
            "summary": "Battery replacement is straightforward.",
            "published_date": "2026-04-19T12:00:00Z",
            "author": {"username": "ifixit_author"},
            "likes": 42,
            "url": "https://www.ifixit.com/Guide/12345",
        },
        guide_id="12345",
        base_url="https://www.ifixit.com",
    )
 
    assert record["repairability_score"] == 8
    assert record["author"] == "ifixit_author"
    assert record["review_text"] == "Battery replacement is straightforward."
 
 
def test_resolve_video_ids_accepts_urls_and_ids():
    assert resolve_video_ids(["https://youtu.be/abcdefghijk", "abcdefghijk"]) == ("abcdefghijk",)
 
 
def test_build_record_uses_real_transcript_segments():
    record = build_record(
        video_id="abcdefghijk",
        segments=[
            {"text": "Great review", "duration": 3.0},
            {"text": "Battery life is strong", "duration": 4.0},
        ],
        metadata={"title": "Sony XM5 Review", "channel": "Tech Reviewer"},
    )
 
    assert record["segment_count"] == 2
    assert record["duration_seconds"] == 7.0
    assert "Battery life is strong" in record["text"]
 
 