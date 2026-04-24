"""S3 storage helpers for run-based pipeline publishing."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
 
from src.common.settings import Settings
 
 
try:
    import boto3
    from boto3.s3.transfer import TransferConfig
except ImportError:
    boto3 = None
    TransferConfig = None
 
 
def sanitize_storage_component(value: str) -> str:
    return str(value).strip().strip("/").replace("\\", "/")
 
 
def join_s3_uri(bucket: str, *parts: str, trailing_slash: bool = False) -> str:
    cleaned_parts = [sanitize_storage_component(part) for part in parts if sanitize_storage_component(part)]
    uri = "s3://" + "/".join([sanitize_storage_component(bucket), *cleaned_parts])
    if trailing_slash and not uri.endswith("/"):
        return uri + "/"
    return uri
 
 
def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got: {uri}")
    stripped = uri[len("s3://"):]
    bucket, _, key = stripped.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in S3 URI: {uri}")
    return bucket, key
 
 
@dataclass(frozen=True, slots=True)
class S3PathResolver:
    bucket: str
    raw_prefix: str
    processed_prefix: str
 
    def raw_run_prefix(self, source: str, run_id: str) -> str:
        return join_s3_uri(
            self.bucket,
            self.raw_prefix,
            sanitize_storage_component(source),
            "runs",
            sanitize_storage_component(run_id),
            trailing_slash=True,
        )
 
    def raw_current_prefix(self, source: str) -> str:
        return join_s3_uri(
            self.bucket,
            self.raw_prefix,
            sanitize_storage_component(source),
            "current",
            trailing_slash=True,
        )
 
    def processed_run_prefix(self, stage: str, run_id: str) -> str:
        return join_s3_uri(
            self.bucket,
            self.processed_prefix,
            sanitize_storage_component(stage),
            "runs",
            sanitize_storage_component(run_id),
            trailing_slash=True,
        )
 
    def processed_current_prefix(self, stage: str) -> str:
        return join_s3_uri(
            self.bucket,
            self.processed_prefix,
            sanitize_storage_component(stage),
            "current",
            trailing_slash=True,
        )
 
    def marker_uri(self, current_prefix: str) -> str:
        return current_prefix.rstrip("/") + "/_LATEST_RUN.json"
 
 
class S3StorageManager:
    """Uploads run artifacts and promotes successful outputs into current/."""
 
    def __init__(self, resolver: S3PathResolver, client: Any) -> None:
        self.resolver = resolver
        self.client = client
 
    @classmethod
    def from_settings(cls, settings: Settings, client: Any | None = None) -> "S3StorageManager":
        if not settings.s3_enabled:
            raise RuntimeError("S3_BUCKET_NAME must be configured for pipeline runs.")
        if client is None:
            if boto3 is None:
                raise RuntimeError("Missing dependency: boto3. Install it before running S3-backed pipelines.")
            client = boto3.client("s3", region_name=settings.aws_region)
        return cls(
            resolver=S3PathResolver(
                bucket=settings.s3_bucket_name,
                raw_prefix=settings.s3_raw_prefix,
                processed_prefix=settings.s3_processed_prefix,
            ),
            client=client,
        )
 
    def upload_file(self, local_path: Path, destination_uri: str) -> str:
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
 
        bucket, key = parse_s3_uri(destination_uri)
        self.client.upload_file(str(local_path), bucket, key)
        return destination_uri

    def copy_object(self, source_uri: str, destination_uri: str) -> str:
        source_bucket, source_key = parse_s3_uri(source_uri)
        destination_bucket, destination_key = parse_s3_uri(destination_uri)
        managed_copy = getattr(self.client, "copy", None)
        if callable(managed_copy):
            config = (
                TransferConfig(
                    multipart_threshold=64 * 1024 * 1024,
                    multipart_chunksize=64 * 1024 * 1024,
                )
                if TransferConfig is not None
                else None
            )
            kwargs: dict[str, Any] = {
                "CopySource": {"Bucket": source_bucket, "Key": source_key},
                "Bucket": destination_bucket,
                "Key": destination_key,
            }
            if config is not None:
                kwargs["Config"] = config
            managed_copy(**kwargs)
        else:
            self.client.copy_object(
                Bucket=destination_bucket,
                Key=destination_key,
                CopySource={"Bucket": source_bucket, "Key": source_key},
            )
        return destination_uri

    def download_file(self, source_uri: str, local_path: Path) -> Path:
        bucket, key = parse_s3_uri(source_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(local_path))
        return local_path

    def download_prefix(
        self,
        source_prefix_uri: str,
        local_dir: Path,
        *,
        clear_destination: bool = False,
        exclude_latest_marker: bool = True,
    ) -> list[Path]:
        """Download an S3 prefix into a local directory using relative keys."""
        source_bucket, source_prefix = parse_s3_uri(source_prefix_uri)
        source_prefix = source_prefix.rstrip("/") + "/"

        download_plan: list[tuple[str, Path]] = []
        for source_uri in self.list_objects(source_prefix_uri):
            bucket, source_key = parse_s3_uri(source_uri)
            if bucket != source_bucket or not source_key.startswith(source_prefix):
                continue

            relative_key = source_key[len(source_prefix):].lstrip("/")
            if not relative_key or relative_key.endswith("/"):
                continue
            if exclude_latest_marker and relative_key == "_LATEST_RUN.json":
                continue

            parts = tuple(part for part in relative_key.replace("\\", "/").split("/") if part)
            if any(part in {".", ".."} for part in parts):
                raise ValueError(f"Unsafe S3 key relative path: {relative_key}")

            local_path = local_dir.joinpath(*parts)
            download_plan.append((source_uri, local_path))

        if not download_plan:
            return []

        if clear_destination and local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths: list[Path] = []
        for source_uri, local_path in download_plan:
            self.download_file(source_uri, local_path)
            downloaded_paths.append(local_path)

        return downloaded_paths

    def count_lines(self, source_uri: str, *, chunk_size: int = 1024 * 1024) -> int:
        bucket, key = parse_s3_uri(source_uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        line_count = 0
        saw_bytes = False
        ended_with_newline = False

        try:
            for chunk in body.iter_chunks(chunk_size=chunk_size):
                if not chunk:
                    continue
                saw_bytes = True
                line_count += chunk.count(b"\n")
                ended_with_newline = chunk.endswith(b"\n")
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

        if saw_bytes and not ended_with_newline:
            line_count += 1
        return line_count
 
    def upload_directory(self, local_dir: Path, destination_prefix: str) -> list[str]:
        if not local_dir.exists() or not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
 
        uploaded_uris: list[str] = []
        for local_path in sorted(path for path in local_dir.rglob("*") if path.is_file()):
            relative_path = local_path.relative_to(local_dir).as_posix()
            destination_uri = destination_prefix + relative_path
            self.upload_file(local_path, destination_uri)
            uploaded_uris.append(destination_uri)
 
        if not uploaded_uris:
            raise RuntimeError(f"No files found to upload in directory: {local_dir}")
        return uploaded_uris
 
    def list_objects(self, prefix_uri: str) -> list[str]:
        bucket, prefix = parse_s3_uri(prefix_uri)
        continuation_token: str | None = None
        objects: list[str] = []
 
        while True:
            params: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation_token:
                params["ContinuationToken"] = continuation_token
 
            response = self.client.list_objects_v2(**params)
            for item in response.get("Contents", []):
                objects.append(join_s3_uri(bucket, item["Key"]))
 
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
 
        return objects
 
    def delete_prefix(self, prefix_uri: str) -> int:
        object_uris = self.list_objects(prefix_uri)
        if not object_uris:
            return 0
 
        bucket, _ = parse_s3_uri(prefix_uri)
        for index in range(0, len(object_uris), 1000):
            batch = object_uris[index:index + 1000]
            self.client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [
                        {"Key": parse_s3_uri(uri)[1]}
                        for uri in batch
                    ]
                },
            )
        return len(object_uris)
 
    def write_json(self, destination_uri: str, payload: dict[str, Any]) -> str:
        bucket, key = parse_s3_uri(destination_uri)
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        return destination_uri
 
    def promote_run_prefix(
        self,
        run_prefix_uri: str,
        current_prefix_uri: str,
        *,
        run_id: str,
        metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
        progress_interval: int = 100,
    ) -> dict[str, Any]:
        source_objects = self.list_objects(run_prefix_uri)
        if not source_objects:
            raise RuntimeError(f"No S3 objects found under run prefix: {run_prefix_uri}")

        total_objects = len(source_objects)
        copied_count = 0

        source_bucket, source_prefix = parse_s3_uri(run_prefix_uri)
        current_bucket, current_prefix = parse_s3_uri(current_prefix_uri)
        marker_uri = self.resolver.marker_uri(current_prefix_uri)
        desired_current_uris: set[str] = set()

        for source_uri in source_objects:
            _, source_key = parse_s3_uri(source_uri)
            relative_key = source_key[len(source_prefix):].lstrip("/")
            destination_key = "/".join(
                part for part in [current_prefix.rstrip("/"), relative_key] if part
            )
            desired_current_uris.add(join_s3_uri(current_bucket, destination_key))

        stale_current_objects = [
            uri for uri in self.list_objects(current_prefix_uri)
            if uri != marker_uri and uri not in desired_current_uris
        ]
        removed_count = len(stale_current_objects)

        for source_uri in source_objects:
            _, source_key = parse_s3_uri(source_uri)
            relative_key = source_key[len(source_prefix):].lstrip("/")
            destination_key = "/".join(
                part for part in [current_prefix.rstrip("/"), relative_key] if part
            )
            self.copy_object(source_uri, join_s3_uri(current_bucket, destination_key))
            copied_count += 1
            if progress_callback and (
                copied_count == total_objects
                or copied_count == 1
                or copied_count % max(1, progress_interval) == 0
            ):
                progress_callback(
                    {
                        "copied_count": copied_count,
                        "removed_count": removed_count,
                        "total_objects": total_objects,
                    }
                )

        if stale_current_objects:
            bucket, _ = parse_s3_uri(current_prefix_uri)
            for index in range(0, len(stale_current_objects), 1000):
                batch = stale_current_objects[index:index + 1000]
                self.client.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [
                            {"Key": parse_s3_uri(uri)[1]}
                            for uri in batch
                        ]
                    },
                )

        marker_payload = {
            "run_id": run_id,
            "promoted_at": datetime.now(UTC).isoformat(),
            **(metadata or {}),
        }
        marker_uri = self.write_json(marker_uri, marker_payload)
        return {
            "copied_count": copied_count,
            "removed_count": removed_count,
            "marker_uri": marker_uri,
        }
 
 
