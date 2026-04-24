"""Unit tests for run context, S3 storage, and structured logging helpers."""
 
from __future__ import annotations
 
import json
import logging
import shutil
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
 
from src.common.run_context import build_run_context, generate_run_id
from src.common.storage import S3PathResolver, S3StorageManager
from src.common.structured_logging import configure_structured_logging, log_event
 
 
class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    class _Body:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def iter_chunks(self, chunk_size: int = 1024 * 1024):
            for offset in range(0, len(self.payload), chunk_size):
                yield self.payload[offset:offset + chunk_size]

        def close(self) -> None:
            return None
 
    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.objects[(bucket, key)])
 
    def list_objects_v2(self, Bucket: str, Prefix: str, ContinuationToken: str | None = None) -> dict:
        matching_keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        return {
            "Contents": [{"Key": key} for key in matching_keys],
            "IsTruncated": False,
        }
 
    def delete_objects(self, Bucket: str, Delete: dict) -> None:
        for item in Delete.get("Objects", []):
            self.objects.pop((Bucket, item["Key"]), None)
 
    def copy_object(self, Bucket: str, Key: str, CopySource: dict) -> None:
        source_bucket = CopySource["Bucket"]
        source_key = CopySource["Key"]
        self.objects[(Bucket, Key)] = self.objects[(source_bucket, source_key)]
 
    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        payload = self.objects[(Bucket, Key)]
        return {
            "Body": self._Body(payload),
            "ContentLength": len(payload),
        }
 
 
def test_generate_run_id_includes_utc_timestamp():
    run_id = generate_run_id(now=datetime(2026, 4, 19, 15, 30, 0, tzinfo=UTC))
    assert run_id.startswith("run_20260419T153000Z_")
    assert len(run_id.split("_")[-1]) == 8
 
 
def test_build_run_context_uses_airflow_env(monkeypatch):
    monkeypatch.setenv("REVIEWPULSE_RUN_ID", "scheduled__2026-04-19T02:00:00+00:00")
    monkeypatch.setenv("REVIEWPULSE_DAG_ID", "reviewpulse_pipeline")
    monkeypatch.setenv("REVIEWPULSE_TASK_ID", "normalize_reviews_spark")
 
    context = build_run_context(stage="normalize reviews spark")
 
    assert context.run_id == "scheduled__2026-04-19T02-00-00-00-00"
    assert context.dag_id == "reviewpulse_pipeline"
    assert context.task_id == "normalize_reviews_spark"
    assert context.stage == "normalize-reviews-spark"
 
 
def test_s3_path_resolver_matches_run_and_current_convention():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
 
    assert resolver.raw_run_prefix("amazon", "run_123") == "s3://reviewpulse-bucket/raw/amazon/runs/run_123/"
    assert resolver.raw_current_prefix("amazon") == "s3://reviewpulse-bucket/raw/amazon/current/"
    assert resolver.processed_run_prefix("normalized_parquet", "run_123") == (
        "s3://reviewpulse-bucket/processed/normalized_parquet/runs/run_123/"
    )
    assert resolver.processed_current_prefix("normalized_parquet") == (
        "s3://reviewpulse-bucket/processed/normalized_parquet/current/"
    )
 
 
def test_promote_run_prefix_replaces_current_and_writes_latest_marker():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)
 
    temp_dir = Path(__file__).resolve().parent / "_tmp_storage_foundation"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_file = temp_dir / "normalized_reviews.jsonl"
        local_file.write_text('{"review_id": "1"}\n', encoding="utf-8")
 
        run_prefix = resolver.processed_run_prefix("normalized_jsonl", "run_123")
        current_prefix = resolver.processed_current_prefix("normalized_jsonl")
 
        manager.upload_file(local_file, run_prefix + local_file.name)
        client.objects[("reviewpulse-bucket", "processed/normalized_jsonl/current/stale.jsonl")] = b"stale"
 
        result = manager.promote_run_prefix(
            run_prefix,
            current_prefix,
            run_id="run_123",
            metadata={"stage": "normalized_jsonl", "status": "success"},
        )
 
        assert result["copied_count"] == 1
        assert ("reviewpulse-bucket", "processed/normalized_jsonl/current/stale.jsonl") not in client.objects
        assert (
            "reviewpulse-bucket",
            "processed/normalized_jsonl/current/normalized_reviews.jsonl",
        ) in client.objects
 
        marker_key = "processed/normalized_jsonl/current/_LATEST_RUN.json"
        marker_payload = json.loads(client.objects[("reviewpulse-bucket", marker_key)].decode("utf-8"))
        assert marker_payload["run_id"] == "run_123"
        assert marker_payload["stage"] == "normalized_jsonl"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_file_fetches_s3_object_to_local_path():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)

    temp_dir = Path(__file__).resolve().parent / "_tmp_storage_download"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_uri = "s3://reviewpulse-bucket/raw/yelp/current/yelp_academic_dataset_review.json"
        client.objects[("reviewpulse-bucket", "raw/yelp/current/yelp_academic_dataset_review.json")] = (
            b'{"review_id":"1"}\n'
        )

        local_path = temp_dir / "yelp_academic_dataset_review.json"
        downloaded = manager.download_file(source_uri, local_path)

        assert downloaded == local_path
        assert local_path.read_text(encoding="utf-8") == '{"review_id":"1"}\n'
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_prefix_syncs_relative_files_and_skips_latest_marker():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)

    temp_dir = Path(__file__).resolve().parent / "_tmp_storage_prefix_download"
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        client.objects[("reviewpulse-bucket", "raw/amazon/current/_LATEST_RUN.json")] = b"{}"
        client.objects[("reviewpulse-bucket", "raw/amazon/current/batch_1.jsonl")] = b'{"id":1}\n'
        client.objects[("reviewpulse-bucket", "raw/amazon/current/nested/batch_2.jsonl")] = b'{"id":2}\n'

        downloaded = manager.download_prefix(
            "s3://reviewpulse-bucket/raw/amazon/current/",
            temp_dir,
            clear_destination=True,
        )

        assert sorted(path.relative_to(temp_dir).as_posix() for path in downloaded) == [
            "batch_1.jsonl",
            "nested/batch_2.jsonl",
        ]
        assert (temp_dir / "batch_1.jsonl").read_text(encoding="utf-8") == '{"id":1}\n'
        assert not (temp_dir / "_LATEST_RUN.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_download_prefix_does_not_clear_destination_when_prefix_is_empty():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)

    temp_dir = Path(__file__).resolve().parent / "_tmp_storage_empty_prefix"
    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        temp_dir.mkdir(parents=True)
        stale_file = temp_dir / "local_only.jsonl"
        stale_file.write_text('{"id": "local"}\n', encoding="utf-8")

        downloaded = manager.download_prefix(
            "s3://reviewpulse-bucket/raw/missing/current/",
            temp_dir,
            clear_destination=True,
        )

        assert downloaded == []
        assert stale_file.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_count_lines_streams_s3_object():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)

    client.objects[("reviewpulse-bucket", "raw/yelp/current/yelp_academic_dataset_review.json")] = (
        b'{"review_id":"1"}\n{"review_id":"2"}\n'
    )

    assert manager.count_lines("s3://reviewpulse-bucket/raw/yelp/current/yelp_academic_dataset_review.json") == 2


def test_promote_run_prefix_reports_progress():
    resolver = S3PathResolver(bucket="reviewpulse-bucket", raw_prefix="raw", processed_prefix="processed")
    client = FakeS3Client()
    manager = S3StorageManager(resolver, client)

    temp_dir = Path(__file__).resolve().parent / "_tmp_storage_foundation_progress"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_prefix = resolver.raw_run_prefix("amazon", "run_123")
        current_prefix = resolver.raw_current_prefix("amazon")

        local_files = []
        for index in range(3):
            local_file = temp_dir / f"batch_{index + 1}.jsonl"
            local_file.write_text(f'{{"review_id": "{index + 1}"}}\n', encoding="utf-8")
            local_files.append(local_file)
            manager.upload_file(local_file, run_prefix + local_file.name)

        progress_events: list[dict[str, int]] = []
        result = manager.promote_run_prefix(
            run_prefix,
            current_prefix,
            run_id="run_123",
            metadata={"source": "amazon", "stage": "ingest_amazon", "status": "success"},
            progress_callback=progress_events.append,
            progress_interval=2,
        )

        assert result["copied_count"] == 3
        assert progress_events == [
            {"copied_count": 1, "removed_count": 0, "total_objects": 3},
            {"copied_count": 2, "removed_count": 0, "total_objects": 3},
            {"copied_count": 3, "removed_count": 0, "total_objects": 3},
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
 
 
def test_structured_logging_emits_json_events():
    stream = StringIO()
    configure_structured_logging("INFO", logger_name="reviewpulse.tests", stream=stream)
    logger = logging.getLogger("reviewpulse.tests.storage")
 
    log_event(
        logger,
        "pipeline_run_started",
        run_id="run_123",
        stage="normalize_local_preview",
        status="started",
    )
 
    payload = json.loads(stream.getvalue().strip())
    assert payload["event_name"] == "pipeline_run_started"
    assert payload["run_id"] == "run_123"
    assert payload["stage"] == "normalize_local_preview"
    assert payload["status"] == "started"
    assert payload["level"] == "info"
 
 
