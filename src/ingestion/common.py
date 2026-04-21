"""Shared helpers for real source ingestion runners."""
 
from __future__ import annotations
 
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
 
from src.common.run_context import PipelineRunContext
from src.common.settings import Settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import log_event
 
 
@dataclass(frozen=True, slots=True)
class SourceIngestionResult:
    source: str
    run_id: str
    record_count: int
    file_count: int
    local_paths: tuple[Path, ...]
    run_prefix: str
    current_prefix: str
 
 
def build_output_path(settings: Settings, source: str, filename: str) -> Path:
    return (settings.data_dir / source / filename).resolve()
 
 
def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return path
 
 
def copy_file(source_path: Path, destination_path: Path) -> Path:
    source_resolved = source_path.resolve()
    destination_resolved = destination_path.resolve()
    destination_resolved.parent.mkdir(parents=True, exist_ok=True)
    if source_resolved == destination_resolved:
        return destination_resolved
    shutil.copy2(source_resolved, destination_resolved)
    return destination_resolved
 
 
def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
 
 
def publish_source_files(
    *,
    source: str,
    local_paths: Sequence[Path],
    record_count: int,
    storage_manager: S3StorageManager,
    run_context: PipelineRunContext,
    logger: Any,
) -> SourceIngestionResult:
    if not local_paths:
        raise RuntimeError(f"No local source files were produced for {source}.")
 
    run_prefix = storage_manager.resolver.raw_run_prefix(source, run_context.run_id)
    current_prefix = storage_manager.resolver.raw_current_prefix(source)
 
    for local_path in local_paths:
        destination_uri = run_prefix + local_path.name
        log_event(
            logger,
            "s3_upload_started",
            source=source,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(local_path),
            output_path=destination_uri,
            status="started",
        )
        storage_manager.upload_file(local_path, destination_uri)
        log_event(
            logger,
            "s3_upload_completed",
            source=source,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(local_path),
            output_path=destination_uri,
            file_count=1,
            status="success",
        )
 
    promotion = storage_manager.promote_run_prefix(
        run_prefix,
        current_prefix,
        run_id=run_context.run_id,
        metadata={"source": source, "stage": run_context.stage, "status": "success"},
    )
    log_event(
        logger,
        "latest_run_promoted",
        source=source,
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        output_path=current_prefix,
        file_count=promotion["copied_count"],
        record_count=record_count,
        status="success",
    )
 
    return SourceIngestionResult(
        source=source,
        run_id=run_context.run_id,
        record_count=record_count,
        file_count=len(local_paths),
        local_paths=tuple(local_paths),
        run_prefix=run_prefix,
        current_prefix=current_prefix,
    )
 
 