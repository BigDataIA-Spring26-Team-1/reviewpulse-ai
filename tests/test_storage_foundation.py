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
 
    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()
 
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
 
 