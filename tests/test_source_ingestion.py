from __future__ import annotations

import json
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
from src.ingestion.amazon import build_dataset_config, run as run_amazon, stream_amazon_records
from src.ingestion.common import count_lines
from src.ingestion.ebay import map_item_summary, run as run_ebay
from src.ingestion.ifixit import map_guide_payload
from src.ingestion.youtube import build_record, resolve_video_ids
from src.ingestion.yelp import (
    BUSINESS_FILENAME,
    REVIEW_FILENAME,
    resolve_yelp_source_files,
    resolve_yelp_source_uris,
    run as run_yelp,
)
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
        amazon_batch_size=2,
        amazon_max_records=0,
        yelp_dataset_path=None,
        yelp_dataset_s3_uri=None,
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


def build_storage_manager(client: FakeS3Client | None = None) -> S3StorageManager:
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    return S3StorageManager(resolver, client or FakeS3Client())


def build_logger() -> logging.Logger:
    logger, _ = build_logger_with_stream()
    return logger


def build_logger_with_stream(name: str = "reviewpulse.tests.ingestion") -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    configure_structured_logging("INFO", logger_name=name, stream=stream)
    return logging.getLogger(name), stream


class FailOnceOnKeyClient(FakeS3Client):
    def __init__(self, fail_key_suffix: str) -> None:
        super().__init__()
        self.fail_key_suffix = fail_key_suffix
        self.failed = False

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        if not self.failed and key.endswith(self.fail_key_suffix):
            self.failed = True
            raise RuntimeError("simulated upload failure")
        super().upload_file(filename, bucket, key)


class FakeStatefulAmazonStream:
    def __init__(self, records: list[dict[str, object]], *, start_index: int = 0) -> None:
        self._records = records
        self._position = start_index

    def __iter__(self) -> FakeStatefulAmazonStream:
        return self

    def __next__(self) -> dict[str, object]:
        if self._position >= len(self._records):
            raise StopIteration
        record = self._records[self._position]
        self._position += 1
        return record

    def state_dict(self) -> dict[str, int]:
        return {"offset": self._position}


def test_build_dataset_config_normalizes_amazon_category():
    assert build_dataset_config("Electronics") == "raw_review_Electronics"
    assert build_dataset_config("raw_review_Books") == "raw_review_Books"


def test_stream_amazon_records_ignores_legacy_max_record_limit(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("amazon_stream_full")
    try:
        settings = replace(build_test_settings(workspace), amazon_max_records=1)

        monkeypatch.setattr(
            "src.ingestion.amazon._load_dataset_module",
            lambda: (
                lambda *_args, **_kwargs: [
                    {"asin": "B001", "user_id": "u1", "rating": 5.0},
                    {"asin": "B002", "user_id": "u2", "rating": 4.0},
                ]
            ),
        )

        records = list(stream_amazon_records(settings))

        assert len(records) == 2
        assert [record["asin"] for record in records] == ["B001", "B002"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


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


def test_resolve_yelp_source_uris_accepts_prefix_and_review_file():
    prefix_review_uri, prefix_business_uri = resolve_yelp_source_uris("s3://source-bucket/yelp-open-dataset/")
    assert prefix_review_uri == "s3://source-bucket/yelp-open-dataset/yelp_academic_dataset_review.json"
    assert prefix_business_uri == "s3://source-bucket/yelp-open-dataset/yelp_academic_dataset_business.json"

    review_uri, business_uri = resolve_yelp_source_uris(
        "s3://source-bucket/yelp-open-dataset/yelp_academic_dataset_review.json"
    )
    assert review_uri == "s3://source-bucket/yelp-open-dataset/yelp_academic_dataset_review.json"
    assert business_uri == "s3://source-bucket/yelp-open-dataset/yelp_academic_dataset_business.json"


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

        run_dir = workspace / "data" / "yelp" / "runs" / "run_test"
        local_review = run_dir / REVIEW_FILENAME
        local_business = run_dir / BUSINESS_FILENAME
        manifest_path = run_dir / "manifest.json"
        assert not (workspace / "data" / "yelp" / REVIEW_FILENAME).exists()
        assert not (workspace / "data" / "yelp" / BUSINESS_FILENAME).exists()
        assert local_review.exists()
        assert local_business.exists()
        assert manifest_path.exists()
        assert count_lines(local_review) == 2
        assert result.record_count == 2
        assert result.file_count == 3
        assert result.local_paths[-1].name == "manifest.json"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_run_yelp_downloads_from_s3_and_publishes_source_files():
    workspace = make_workspace("run_yelp_s3")
    try:
        storage_manager = build_storage_manager()
        storage_manager.client.objects[
            ("source-bucket", "incoming/yelp/yelp_academic_dataset_review.json")
        ] = b'{"review_id":"1"}\n{"review_id":"2"}\n'
        storage_manager.client.objects[
            ("source-bucket", "incoming/yelp/yelp_academic_dataset_business.json")
        ] = b'{"business_id":"b1"}\n'

        settings = replace(
            build_test_settings(workspace),
            yelp_dataset_s3_uri="s3://source-bucket/incoming/yelp/",
        )
        result = run_yelp(
            settings=settings,
            run_context=build_run_context(stage="ingest_yelp", source="yelp", run_id="run_test"),
            logger=build_logger(),
            storage_manager=storage_manager,
        )

        run_dir = workspace / "data" / "yelp" / "runs" / "run_test"
        local_review = run_dir / REVIEW_FILENAME
        local_business = run_dir / BUSINESS_FILENAME
        manifest_path = run_dir / "manifest.json"
        assert not (workspace / "data" / "yelp" / REVIEW_FILENAME).exists()
        assert not (workspace / "data" / "yelp" / BUSINESS_FILENAME).exists()
        assert not local_review.exists()
        assert not local_business.exists()
        assert manifest_path.exists()
        assert result.record_count == 2
        assert result.file_count == 3
        assert result.local_paths == (manifest_path.resolve(),)
        assert (
            "reviewpulse-bucket",
            "raw/yelp/current/yelp_academic_dataset_review.json",
        ) in storage_manager.client.objects
        assert (
            "reviewpulse-bucket",
            "raw/yelp/current/yelp_academic_dataset_business.json",
        ) in storage_manager.client.objects
        assert (
            "reviewpulse-bucket",
            "raw/yelp/current/manifest.json",
        ) in storage_manager.client.objects
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_run_amazon_writes_batches_manifest_and_promotes(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("run_amazon")
    try:
        settings = build_test_settings(workspace)
        storage_manager = build_storage_manager()
        logger, stream = build_logger_with_stream("reviewpulse.tests.ingestion.amazon")
        records = [
            {"asin": "B001", "user_id": "u1", "rating": 5.0},
            {"asin": "B002", "user_id": "u2", "rating": 4.0},
            {"asin": "B003", "user_id": "u3", "rating": 3.0},
        ]

        monkeypatch.setattr(
            "src.ingestion.amazon.stream_amazon_records",
            lambda _settings, skip_records=0, stream_state=None, logger=None, run_context=None: iter(
                records[skip_records:]
            ),
        )

        result = run_amazon(
            settings=settings,
            run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
            logger=logger,
            storage_manager=storage_manager,
        )

        run_dir = workspace / "data" / "amazon" / "runs" / "run_test"
        batch_paths = sorted(run_dir.glob("amazon_reviews_batch_*.jsonl"))
        manifest_path = run_dir / "manifest.json"
        assert not (workspace / "data" / "amazon" / "amazon_reviews.jsonl").exists()
        assert len(batch_paths) == 2
        assert count_lines(batch_paths[0]) == 2
        assert count_lines(batch_paths[1]) == 1
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["total_batches"] == 2
        assert manifest["total_records"] == 3
        assert result.record_count == 3
        assert result.file_count == 3
        assert result.local_paths[-1].name == "manifest.json"

        current_keys = {
            key
            for bucket, key in storage_manager.client.objects
            if bucket == "reviewpulse-bucket" and key.startswith("raw/amazon/current/")
        }
        assert "raw/amazon/current/amazon_reviews_batch_000001_records_00000002.jsonl" in current_keys
        assert "raw/amazon/current/amazon_reviews_batch_000002_records_00000001.jsonl" in current_keys
        assert "raw/amazon/current/manifest.json" in current_keys

        events = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
        batch_events = [event for event in events if event["event_name"] == "amazon_batch_completed"]
        assert len(batch_events) == 2
        assert batch_events[0]["batch_index"] == 1
        assert batch_events[0]["batch_size"] == 2
        assert batch_events[0]["cumulative_records"] == 2
        assert batch_events[0]["status"] == "success"
        assert batch_events[1]["batch_index"] == 2
        assert batch_events[1]["cumulative_records"] == 3
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_run_amazon_resumes_after_failed_batch_upload(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("run_amazon_resume")
    try:
        settings = build_test_settings(workspace)
        records = [
            {"asin": "B001", "user_id": "u1", "rating": 5.0},
            {"asin": "B002", "user_id": "u2", "rating": 4.0},
            {"asin": "B003", "user_id": "u3", "rating": 3.0},
            {"asin": "B004", "user_id": "u4", "rating": 2.0},
        ]
        stream_calls: list[int] = []

        def fake_stream(
            _settings: Settings,
            *,
            skip_records: int = 0,
            stream_state: dict[str, int] | None = None,
            logger: logging.Logger | None = None,
            run_context=None,
        ):
            stream_calls.append(skip_records)
            return iter(records[skip_records:])

        monkeypatch.setattr("src.ingestion.amazon.stream_amazon_records", fake_stream)

        client = FailOnceOnKeyClient("amazon_reviews_batch_000002_records_00000002.jsonl")
        storage_manager = build_storage_manager(client)
        failure_logger, failure_stream = build_logger_with_stream("reviewpulse.tests.ingestion.amazon.resume.fail")

        with pytest.raises(RuntimeError, match="after successful batch 1 with 2 committed records"):
            run_amazon(
                settings=settings,
                run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
                logger=failure_logger,
                storage_manager=storage_manager,
            )

        run_dir = workspace / "data" / "amazon" / "runs" / "run_test"
        checkpoint_payload = json.loads((run_dir / "_checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint_payload["completed"] is False
        assert checkpoint_payload["cumulative_records"] == 2

        failure_events = [json.loads(line) for line in failure_stream.getvalue().splitlines() if line.strip()]
        failed_batch_event = next(event for event in failure_events if event["event_name"] == "amazon_batch_failed")
        failed_run_event = next(event for event in failure_events if event["event_name"] == "pipeline_run_failed")
        assert failed_batch_event["last_successful_batch_index"] == 1
        assert failed_run_event["last_successful_batch_index"] == 1
        assert failed_run_event["last_successful_cumulative_records"] == 2

        result = run_amazon(
            settings=settings,
            run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
            logger=build_logger(),
            storage_manager=storage_manager,
        )

        manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_payload = json.loads((run_dir / "_checkpoint.json").read_text(encoding="utf-8"))
        assert stream_calls == [0, 2]
        assert sorted(path.name for path in run_dir.glob("amazon_reviews_batch_*.jsonl")) == [
            "amazon_reviews_batch_000001_records_00000002.jsonl",
            "amazon_reviews_batch_000002_records_00000002.jsonl",
        ]
        assert manifest_payload["total_batches"] == 2
        assert manifest_payload["total_records"] == 4
        assert checkpoint_payload["completed"] is True
        assert result.record_count == 4
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_run_amazon_resumes_from_stream_state_without_replay(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("run_amazon_stream_state_resume")
    try:
        settings = build_test_settings(workspace)
        records = [
            {"asin": "B001", "user_id": "u1", "rating": 5.0},
            {"asin": "B002", "user_id": "u2", "rating": 4.0},
            {"asin": "B003", "user_id": "u3", "rating": 3.0},
            {"asin": "B004", "user_id": "u4", "rating": 2.0},
        ]
        stream_calls: list[tuple[int, dict[str, int] | None]] = []

        def fake_stream(
            _settings: Settings,
            *,
            skip_records: int = 0,
            stream_state: dict[str, int] | None = None,
            logger: logging.Logger | None = None,
            run_context=None,
        ):
            stream_calls.append((skip_records, stream_state))
            start_index = stream_state["offset"] if stream_state is not None else skip_records
            return FakeStatefulAmazonStream(records, start_index=start_index)

        monkeypatch.setattr("src.ingestion.amazon.stream_amazon_records", fake_stream)

        client = FailOnceOnKeyClient("amazon_reviews_batch_000002_records_00000002.jsonl")
        storage_manager = build_storage_manager(client)

        with pytest.raises(RuntimeError, match="after successful batch 1 with 2 committed records"):
            run_amazon(
                settings=settings,
                run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
                logger=build_logger(),
                storage_manager=storage_manager,
            )

        run_dir = workspace / "data" / "amazon" / "runs" / "run_test"
        checkpoint_payload = json.loads((run_dir / "_checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint_payload["stream_state"] == {"offset": 2}

        result = run_amazon(
            settings=settings,
            run_context=build_run_context(stage="ingest_amazon", source="amazon", run_id="run_test"),
            logger=build_logger(),
            storage_manager=storage_manager,
        )

        assert result.record_count == 4
        assert stream_calls == [
            (0, None),
            (2, {"offset": 2}),
        ]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_run_amazon_reuses_latest_incomplete_run_without_duplicate_batches(monkeypatch: pytest.MonkeyPatch):
    workspace = make_workspace("run_amazon_auto_resume")
    try:
        settings = build_test_settings(workspace)
        storage_manager = build_storage_manager()
        run_dir = workspace / "data" / "amazon" / "runs" / "run_incomplete"
        run_dir.mkdir(parents=True, exist_ok=True)

        batch_one_path = run_dir / "amazon_reviews_batch_000001_records_00000002.jsonl"
        batch_one_path.write_text(
            '{"asin": "B001", "user_id": "u1", "rating": 5.0}\n'
            '{"asin": "B002", "user_id": "u2", "rating": 4.0}\n',
            encoding="utf-8",
        )
        (run_dir / "_checkpoint.json").write_text(
            json.dumps(
                {
                    "completed": False,
                    "cumulative_records": 2,
                    "last_successful_batch_index": 1,
                    "last_successful_batch_size": 2,
                    "last_successful_output_path": (
                        "s3://reviewpulse-bucket/raw/amazon/runs/run_incomplete/"
                        "amazon_reviews_batch_000001_records_00000002.jsonl"
                    ),
                    "total_batches": 1,
                }
            ),
            encoding="utf-8",
        )
        storage_manager.upload_file(
            batch_one_path,
            "s3://reviewpulse-bucket/raw/amazon/runs/run_incomplete/amazon_reviews_batch_000001_records_00000002.jsonl",
        )

        monkeypatch.delenv("REVIEWPULSE_RUN_ID", raising=False)
        monkeypatch.delenv("AIRFLOW_CTX_RUN_ID", raising=False)
        monkeypatch.delenv("AIRFLOW_CTX_DAG_RUN_ID", raising=False)

        records = [
            {"asin": "B001", "user_id": "u1", "rating": 5.0},
            {"asin": "B002", "user_id": "u2", "rating": 4.0},
            {"asin": "B003", "user_id": "u3", "rating": 3.0},
            {"asin": "B004", "user_id": "u4", "rating": 2.0},
        ]
        stream_calls: list[int] = []

        def fake_stream(
            _settings: Settings,
            *,
            skip_records: int = 0,
            stream_state: dict[str, int] | None = None,
            logger: logging.Logger | None = None,
            run_context=None,
        ):
            stream_calls.append(skip_records)
            return iter(records[skip_records:])

        monkeypatch.setattr("src.ingestion.amazon.stream_amazon_records", fake_stream)

        result = run_amazon(
            settings=settings,
            logger=build_logger(),
            storage_manager=storage_manager,
        )

        manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert result.run_id == "run_incomplete"
        assert result.record_count == 4
        assert stream_calls == [2]
        assert manifest_payload["total_batches"] == 2
        assert manifest_payload["total_records"] == 4
        assert sorted(path.name for path in run_dir.glob("amazon_reviews_batch_*.jsonl")) == [
            "amazon_reviews_batch_000001_records_00000002.jsonl",
            "amazon_reviews_batch_000002_records_00000002.jsonl",
        ]
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
