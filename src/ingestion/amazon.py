"""Amazon Reviews 2023 ingestion runner."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import (
    SourceIngestionResult,
    promote_source_run,
    upload_source_file,
    write_json,
    write_jsonl,
)


SOURCE_NAME = "amazon"
RUNS_DIRNAME = "runs"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_FILENAME = "_checkpoint.json"
BATCH_FILENAME_TEMPLATE = "amazon_reviews_batch_{batch_index:06d}_records_{batch_size:08d}.jsonl"
LEGACY_RUN_SNAPSHOT_FILENAME = "amazon_reviews.jsonl"
RESUME_PROGRESS_LOG_INTERVAL = 500_000
RUN_ID_ENV_VARS = (
    "REVIEWPULSE_RUN_ID",
    "AIRFLOW_CTX_RUN_ID",
    "AIRFLOW_CTX_DAG_RUN_ID",
)


def build_dataset_config(category: str) -> str:
    normalized = str(category).strip().replace(" ", "_")
    if not normalized:
        raise RuntimeError("AMAZON_CATEGORY must be configured.")
    if normalized.startswith("raw_review_"):
        return normalized
    return f"raw_review_{normalized}"


def _load_dataset_module() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `datasets`. Run `poetry install` so Amazon ingestion can stream Hugging Face data."
        ) from exc
    return load_dataset


def _run_directory(settings: Settings, run_id: str) -> Path:
    return (settings.data_dir / SOURCE_NAME / RUNS_DIRNAME / run_id).resolve()


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_FILENAME


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_FILENAME


def _legacy_run_snapshot_path(run_dir: Path) -> Path:
    return run_dir / LEGACY_RUN_SNAPSHOT_FILENAME


def _batch_filename(batch_index: int, batch_size: int) -> str:
    return BATCH_FILENAME_TEMPLATE.format(batch_index=batch_index, batch_size=batch_size)


def _default_checkpoint_state() -> dict[str, Any]:
    return {
        "completed": False,
        "cumulative_records": 0,
        "last_successful_batch_index": 0,
        "last_successful_batch_size": 0,
        "last_successful_output_path": None,
        "stream_state": None,
        "total_batches": 0,
    }


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    state = _default_checkpoint_state()
    if not path.exists():
        return state

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Amazon checkpoint is not valid JSON: {path}") from exc

    state["completed"] = bool(payload.get("completed", False))
    state["cumulative_records"] = max(0, int(payload.get("cumulative_records", 0) or 0))
    state["last_successful_batch_index"] = max(0, int(payload.get("last_successful_batch_index", 0) or 0))
    state["last_successful_batch_size"] = max(0, int(payload.get("last_successful_batch_size", 0) or 0))
    state["last_successful_output_path"] = payload.get("last_successful_output_path")
    state["stream_state"] = payload.get("stream_state")
    state["total_batches"] = max(
        state["last_successful_batch_index"],
        int(payload.get("total_batches", state["last_successful_batch_index"]) or 0),
    )
    return state


def _has_explicit_run_id() -> bool:
    return any(os.getenv(name, "").strip() for name in RUN_ID_ENV_VARS)


def _discover_latest_incomplete_run_id(settings: Settings) -> str | None:
    runs_dir = (settings.data_dir / SOURCE_NAME / RUNS_DIRNAME).resolve()
    if not runs_dir.exists() or not runs_dir.is_dir():
        return None

    latest_candidate: tuple[float, str] | None = None
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        checkpoint_path = _checkpoint_path(run_dir)
        if not checkpoint_path.exists():
            continue

        checkpoint_state = _load_checkpoint_state(checkpoint_path)
        if checkpoint_state["completed"]:
            continue

        sort_key = max(run_dir.stat().st_mtime, checkpoint_path.stat().st_mtime)
        candidate = (sort_key, run_dir.name)
        if latest_candidate is None or candidate[0] > latest_candidate[0]:
            latest_candidate = candidate

    return latest_candidate[1] if latest_candidate else None


def _resolve_successful_batch_paths(run_dir: Path, total_batches: int) -> tuple[Path, ...]:
    batch_paths: list[Path] = []
    for batch_index in range(1, total_batches + 1):
        matches = sorted(run_dir.glob(f"amazon_reviews_batch_{batch_index:06d}_records_*.jsonl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Amazon resume state expected one local batch file for batch {batch_index} in {run_dir}."
            )
        batch_paths.append(matches[0].resolve())
    return tuple(batch_paths)


class AmazonRecordStream:
    def __init__(
        self,
        *,
        dataset: Any,
        iterator: Iterator[dict[str, Any]],
        resume_strategy: str,
    ) -> None:
        self._dataset = dataset
        self._iterator = iterator
        self.resume_strategy = resume_strategy

    def __iter__(self) -> AmazonRecordStream:
        return self

    def __next__(self) -> dict[str, Any]:
        return next(self._iterator)

    def state_dict(self) -> dict[str, Any] | None:
        if hasattr(self._dataset, "state_dict"):
            return self._dataset.state_dict()
        return None


def _load_streaming_dataset(settings: Settings) -> Any:
    if settings.amazon_batch_size <= 0:
        raise RuntimeError("AMAZON_BATCH_SIZE must be greater than zero.")

    load_dataset = _load_dataset_module()
    dataset_config = build_dataset_config(settings.amazon_category)
    return load_dataset(
        settings.amazon_dataset_name,
        name=dataset_config,
        split="full",
        streaming=True,
        trust_remote_code=True,
        token=settings.huggingface_token or None,
    )


def stream_amazon_records(
    settings: Settings,
    *,
    skip_records: int = 0,
    stream_state: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    run_context: PipelineRunContext | None = None,
) -> AmazonRecordStream:
    dataset = _load_streaming_dataset(settings)
    resume_strategy = "fresh"

    if stream_state is not None and hasattr(dataset, "load_state_dict"):
        dataset.load_state_dict(stream_state)
        skip_records = 0
        resume_strategy = "stream_state"
    elif skip_records > 0:
        resume_strategy = "replay_skip"

    def _iter_rows() -> Iterator[dict[str, Any]]:
        remaining_skip = skip_records
        skipped_records = 0
        skip_started_at: float | None = None
        next_progress_log_at = RESUME_PROGRESS_LOG_INTERVAL

        if remaining_skip > 0:
            skip_started_at = time.perf_counter()
            if logger is not None and run_context is not None:
                log_event(
                    logger,
                    "source_fetch_resume_replay_started",
                    **run_context.as_log_fields(),
                    replay_target_records=remaining_skip,
                    status="started",
                )

        for row in dataset:
            if remaining_skip > 0:
                remaining_skip -= 1
                skipped_records += 1
                if (
                    logger is not None
                    and run_context is not None
                    and skipped_records >= next_progress_log_at
                ):
                    log_event(
                        logger,
                        "source_fetch_resume_replay_progress",
                        **run_context.as_log_fields(),
                        replayed_records=skipped_records,
                        remaining_records=remaining_skip,
                        status="running",
                    )
                    next_progress_log_at += RESUME_PROGRESS_LOG_INTERVAL
                if remaining_skip == 0 and logger is not None and run_context is not None:
                    log_event(
                        logger,
                        "source_fetch_resume_replay_completed",
                        **run_context.as_log_fields(),
                        replayed_records=skipped_records,
                        duration_ms=round((time.perf_counter() - (skip_started_at or time.perf_counter())) * 1000, 2),
                        status="success",
                    )
                continue
            yield dict(row)

        if remaining_skip > 0:
            raise RuntimeError(
                f"Amazon streaming replay ended before checkpoint catch-up: "
                f"{remaining_skip} records still needed."
            )

    return AmazonRecordStream(
        dataset=dataset,
        iterator=_iter_rows(),
        resume_strategy=resume_strategy,
    )


def fetch_amazon_records(settings: Settings) -> list[dict[str, Any]]:
    return list(stream_amazon_records(settings))


def _checkpoint_payload(
    *,
    batch_index: int,
    batch_size: int,
    cumulative_records: int,
    output_path: str,
    stream_state: dict[str, Any] | None,
    completed: bool,
) -> dict[str, Any]:
    return {
        "completed": completed,
        "cumulative_records": cumulative_records,
        "last_successful_batch_index": batch_index,
        "last_successful_batch_size": batch_size,
        "last_successful_output_path": output_path,
        "stream_state": stream_state,
        "total_batches": batch_index,
    }


def _process_batch(
    *,
    batch_index: int,
    batch_records: list[dict[str, Any]],
    run_dir: Path,
    checkpoint_path: Path,
    run_context: PipelineRunContext,
    logger: logging.Logger,
    storage_manager: S3StorageManager,
    run_prefix: str,
    previous_checkpoint: dict[str, Any],
    stream_state: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    batch_size = len(batch_records)
    batch_path = run_dir / _batch_filename(batch_index, batch_size)
    output_path = run_prefix + batch_path.name
    started_at = time.perf_counter()

    try:
        write_jsonl(batch_path, batch_records)
        upload_source_file(
            source=SOURCE_NAME,
            local_path=batch_path,
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
            destination_uri=output_path,
        )
        checkpoint_payload = _checkpoint_payload(
            batch_index=batch_index,
            batch_size=batch_size,
            cumulative_records=previous_checkpoint["cumulative_records"] + batch_size,
            output_path=output_path,
            stream_state=stream_state,
            completed=False,
        )
        write_json(checkpoint_path, checkpoint_payload)
    except Exception as error:
        if batch_path.exists():
            batch_path.unlink(missing_ok=True)
        log_event(
            logger,
            "amazon_batch_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            batch_index=batch_index,
            batch_size=batch_size,
            cumulative_records=previous_checkpoint["cumulative_records"],
            output_path=output_path,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            last_successful_batch_index=previous_checkpoint["last_successful_batch_index"] or None,
            last_successful_output_path=previous_checkpoint["last_successful_output_path"],
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise

    log_event(
        logger,
        "amazon_batch_completed",
        **run_context.as_log_fields(),
        batch_index=batch_index,
        batch_size=batch_size,
        cumulative_records=checkpoint_payload["cumulative_records"],
        output_path=output_path,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
    )
    return batch_path.resolve(), checkpoint_payload


def _build_manifest_payload(
    *,
    settings: Settings,
    run_context: PipelineRunContext,
    dataset_config: str,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "batch_size": settings.amazon_batch_size,
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset_config": dataset_config,
        "dataset_name": settings.amazon_dataset_name,
        "run_id": run_context.run_id,
        "source": SOURCE_NAME,
        "total_batches": checkpoint_state["last_successful_batch_index"],
        "total_records": checkpoint_state["cumulative_records"],
    }


def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    resumed_run_id: str | None = None
    if run_context is None:
        resumed_run_id = None if _has_explicit_run_id() else _discover_latest_incomplete_run_id(settings)
        run_context = build_run_context(stage="ingest_amazon", source=SOURCE_NAME, run_id=resumed_run_id)
    logger = logger or get_logger("ingestion.amazon")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
    dataset_config = build_dataset_config(settings.amazon_category)
    run_dir = _run_directory(settings, run_context.run_id)
    checkpoint_path = _checkpoint_path(run_dir)
    manifest_path = _manifest_path(run_dir)
    legacy_run_snapshot_path = _legacy_run_snapshot_path(run_dir)
    run_prefix = storage_manager.resolver.raw_run_prefix(SOURCE_NAME, run_context.run_id)
    current_prefix = storage_manager.resolver.raw_current_prefix(SOURCE_NAME)
    checkpoint_state = _load_checkpoint_state(checkpoint_path)

    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
    if resumed_run_id is not None:
        log_event(
            logger,
            "existing_incomplete_run_reused",
            **run_context.as_log_fields(),
            status="resumed",
        )

    try:
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
            status="started",
        )

        if legacy_run_snapshot_path.exists():
            legacy_size_bytes = legacy_run_snapshot_path.stat().st_size
            legacy_run_snapshot_path.unlink()
            log_event(
                logger,
                "legacy_local_snapshot_removed",
                **run_context.as_log_fields(),
                input_path=str(legacy_run_snapshot_path),
                file_size_bytes=legacy_size_bytes,
                status="success",
            )

        if checkpoint_state["completed"]:
            batch_paths = _resolve_successful_batch_paths(
                run_dir,
                checkpoint_state["last_successful_batch_index"],
            )
            if not manifest_path.exists():
                raise RuntimeError(f"Amazon run manifest is missing for completed run: {manifest_path}")
            file_count = len(batch_paths) + 1
            log_event(
                logger,
                "source_fetch_completed",
                **run_context.as_log_fields(),
                input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
                output_path=current_prefix,
                record_count=checkpoint_state["cumulative_records"],
                file_count=file_count,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                status="success",
            )
            log_event(
                logger,
                "pipeline_run_completed",
                **run_context.as_log_fields(),
                output_path=current_prefix,
                record_count=checkpoint_state["cumulative_records"],
                file_count=file_count,
                duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                status="success",
            )
            return SourceIngestionResult(
                source=SOURCE_NAME,
                run_id=run_context.run_id,
                record_count=checkpoint_state["cumulative_records"],
                file_count=file_count,
                local_paths=(*batch_paths, manifest_path.resolve()),
                run_prefix=run_prefix,
                current_prefix=current_prefix,
            )

        successful_batch_paths = list(
            _resolve_successful_batch_paths(
                run_dir,
                checkpoint_state["last_successful_batch_index"],
            )
        )

        if checkpoint_state["cumulative_records"] > 0:
            log_event(
                logger,
                "source_fetch_resumed",
                **run_context.as_log_fields(),
                batch_index=checkpoint_state["last_successful_batch_index"],
                cumulative_records=checkpoint_state["cumulative_records"],
                output_path=checkpoint_state["last_successful_output_path"],
                resume_strategy="stream_state" if checkpoint_state["stream_state"] is not None else "replay_skip",
                status="resumed",
            )

        pending_batch: list[dict[str, Any]] = []
        record_stream = stream_amazon_records(
            settings,
            skip_records=checkpoint_state["cumulative_records"],
            stream_state=checkpoint_state["stream_state"],
            logger=logger,
            run_context=run_context,
        )
        for row in record_stream:
            pending_batch.append(row)
            if len(pending_batch) < settings.amazon_batch_size:
                continue
            batch_path, checkpoint_state = _process_batch(
                batch_index=checkpoint_state["last_successful_batch_index"] + 1,
                batch_records=pending_batch,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                run_context=run_context,
                logger=logger,
                storage_manager=storage_manager,
                run_prefix=run_prefix,
                previous_checkpoint=checkpoint_state,
                stream_state=record_stream.state_dict() if hasattr(record_stream, "state_dict") else None,
            )
            successful_batch_paths.append(batch_path)
            pending_batch = []

        if pending_batch:
            batch_path, checkpoint_state = _process_batch(
                batch_index=checkpoint_state["last_successful_batch_index"] + 1,
                batch_records=pending_batch,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                run_context=run_context,
                logger=logger,
                storage_manager=storage_manager,
                run_prefix=run_prefix,
                previous_checkpoint=checkpoint_state,
                stream_state=record_stream.state_dict() if hasattr(record_stream, "state_dict") else None,
            )
            successful_batch_paths.append(batch_path)

        if checkpoint_state["cumulative_records"] <= 0:
            raise RuntimeError("Amazon ingestion returned zero records from Hugging Face streaming.")

        manifest_payload = _build_manifest_payload(
            settings=settings,
            run_context=run_context,
            dataset_config=dataset_config,
            checkpoint_state=checkpoint_state,
        )
        write_json(manifest_path, manifest_payload)
        upload_source_file(
            source=SOURCE_NAME,
            local_path=manifest_path,
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
            destination_uri=run_prefix + manifest_path.name,
        )
        run_prefix, current_prefix = promote_source_run(
            source=SOURCE_NAME,
            record_count=checkpoint_state["cumulative_records"],
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        final_checkpoint = {
            **checkpoint_state,
            "completed": True,
            "total_batches": checkpoint_state["last_successful_batch_index"],
        }
        write_json(checkpoint_path, final_checkpoint)

        file_count = len(successful_batch_paths) + 1
        log_event(
            logger,
            "source_fetch_completed",
            **run_context.as_log_fields(),
            input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
            output_path=current_prefix,
            record_count=checkpoint_state["cumulative_records"],
            file_count=file_count,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=current_prefix,
            record_count=checkpoint_state["cumulative_records"],
            file_count=file_count,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        return SourceIngestionResult(
            source=SOURCE_NAME,
            run_id=run_context.run_id,
            record_count=checkpoint_state["cumulative_records"],
            file_count=file_count,
            local_paths=(*successful_batch_paths, manifest_path.resolve()),
            run_prefix=run_prefix,
            current_prefix=current_prefix,
        )
    except Exception as error:
        last_successful_batch_index = checkpoint_state["last_successful_batch_index"] or None
        last_successful_output_path = checkpoint_state["last_successful_output_path"]
        cumulative_records = checkpoint_state["cumulative_records"]
        log_event(
            logger,
            "source_fetch_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            last_successful_batch_index=last_successful_batch_index,
            last_successful_output_path=last_successful_output_path,
            last_successful_cumulative_records=cumulative_records,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        log_event(
            logger,
            "pipeline_run_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            last_successful_batch_index=last_successful_batch_index,
            last_successful_output_path=last_successful_output_path,
            last_successful_cumulative_records=cumulative_records,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        if cumulative_records > 0:
            raise RuntimeError(
                f"Amazon batch ingestion failed after successful batch {last_successful_batch_index} "
                f"with {cumulative_records} committed records."
            ) from error
        raise RuntimeError("Amazon batch ingestion failed before any batch completed.") from error


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    run()


if __name__ == "__main__":
    main()
