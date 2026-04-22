"""Yelp Open Dataset ingestion runner."""
 
from __future__ import annotations
 
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import (
    SourceIngestionResult,
    copy_source_object,
    copy_file,
    count_lines,
    promote_source_run,
    upload_source_file,
    write_json,
)
 
 
SOURCE_NAME = "yelp"
RUNS_DIRNAME = "runs"
MANIFEST_FILENAME = "manifest.json"
REVIEW_FILENAME = "yelp_academic_dataset_review.json"
BUSINESS_FILENAME = "yelp_academic_dataset_business.json"


def _run_directory(settings: Settings, run_id: str) -> Path:
    return (settings.data_dir / SOURCE_NAME / RUNS_DIRNAME / run_id).resolve()


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_FILENAME
 
 
def resolve_yelp_source_files(dataset_path: Path) -> tuple[Path, Path]:
    resolved = dataset_path.resolve()
    if not resolved.exists():
        raise RuntimeError(f"Configured Yelp dataset path does not exist: {resolved}")
 
    if resolved.is_dir():
        review_path = resolved / REVIEW_FILENAME
        business_path = resolved / BUSINESS_FILENAME
    else:
        parent = resolved.parent
        if resolved.name == REVIEW_FILENAME:
            review_path = resolved
            business_path = parent / BUSINESS_FILENAME
        elif resolved.name == BUSINESS_FILENAME:
            review_path = parent / REVIEW_FILENAME
            business_path = resolved
        else:
            raise RuntimeError(
                "YELP_DATASET_PATH must point to the Yelp dataset directory or one of the official JSON files."
            )
 
    if not review_path.exists():
        raise RuntimeError(f"Missing Yelp review file: {review_path}")
    if not business_path.exists():
        raise RuntimeError(f"Missing Yelp business file: {business_path}")
    return review_path.resolve(), business_path.resolve()


def resolve_yelp_source_uris(dataset_uri: str) -> tuple[str, str]:
    normalized = dataset_uri.strip()
    if not normalized.startswith("s3://"):
        raise RuntimeError(f"Expected S3 URI for Yelp dataset source, got: {dataset_uri}")

    def _join(prefix: str, filename: str) -> str:
        return prefix.rstrip("/") + "/" + filename

    if normalized.endswith("/"):
        prefix = normalized.rstrip("/")
        return (
            _join(prefix, REVIEW_FILENAME),
            _join(prefix, BUSINESS_FILENAME),
        )

    if normalized.endswith(REVIEW_FILENAME):
        base_prefix = normalized[: -len(REVIEW_FILENAME)].rstrip("/")
        return (
            normalized,
            _join(base_prefix, BUSINESS_FILENAME),
        )

    if normalized.endswith(BUSINESS_FILENAME):
        base_prefix = normalized[: -len(BUSINESS_FILENAME)].rstrip("/")
        return (
            _join(base_prefix, REVIEW_FILENAME),
            normalized,
        )

    return (
        _join(normalized, REVIEW_FILENAME),
        _join(normalized, BUSINESS_FILENAME),
    )


def validate_yelp_source_uris(
    *,
    storage_manager: S3StorageManager,
    review_uri: str,
    business_uri: str,
) -> None:
    source_prefix = review_uri.rsplit("/", 1)[0] + "/"
    available_objects = set(storage_manager.list_objects(source_prefix))
    missing_objects = [uri for uri in (review_uri, business_uri) if uri not in available_objects]
    if not missing_objects:
        return

    available_suffixes = sorted(uri[len(source_prefix):] for uri in available_objects)[:20]
    raise RuntimeError(
        "Configured Yelp S3 source is missing required files. "
        f"Expected: {review_uri} and {business_uri}. "
        f"Found under {source_prefix}: {available_suffixes or 'no objects'}"
    )


def stage_yelp_source_files(
    *,
    settings: Settings,
    run_dir: Path,
) -> tuple[Path, Path, str]:
    source_location = settings.yelp_dataset_source
    if source_location is None:
        raise RuntimeError("YELP_DATASET_PATH or YELP_DATASET_S3_URI must be configured for Yelp ingestion.")

    if settings.yelp_dataset_path is None:
        raise RuntimeError("YELP_DATASET_PATH or YELP_DATASET_S3_URI must be configured for Yelp ingestion.")

    review_path, business_path = resolve_yelp_source_files(settings.yelp_dataset_path)
    staged_review_path = copy_file(review_path, run_dir / REVIEW_FILENAME)
    staged_business_path = copy_file(business_path, run_dir / BUSINESS_FILENAME)
    return staged_review_path, staged_business_path, str(settings.yelp_dataset_path)


def _build_manifest_payload(
    *,
    run_context: PipelineRunContext,
    input_location: str,
    review_count: int,
) -> dict[str, object]:
    return {
        "completed_at": datetime.now(UTC).isoformat(),
        "input_location": input_location,
        "run_id": run_context.run_id,
        "source": SOURCE_NAME,
        "total_files": 2,
        "total_records": review_count,
    }
 
 
def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_yelp", source=SOURCE_NAME)
    logger = logger or get_logger("ingestion.yelp")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
    run_dir = _run_directory(settings, run_context.run_id)
    manifest_path = _manifest_path(run_dir)
    run_prefix = storage_manager.resolver.raw_run_prefix(SOURCE_NAME, run_context.run_id)
    current_prefix = storage_manager.resolver.raw_current_prefix(SOURCE_NAME)
 
    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
 
    try:
        if not settings.has_yelp_dataset_source:
            raise RuntimeError("YELP_DATASET_PATH or YELP_DATASET_S3_URI must be configured for Yelp ingestion.")
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=str(settings.yelp_dataset_source),
            status="started",
        )
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        staged_review_path: Path | None = None
        staged_business_path: Path | None = None
        if settings.yelp_dataset_s3_uri:
            review_uri, business_uri = resolve_yelp_source_uris(settings.yelp_dataset_s3_uri)
            validate_yelp_source_uris(
                storage_manager=storage_manager,
                review_uri=review_uri,
                business_uri=business_uri,
            )
            input_location = settings.yelp_dataset_s3_uri
            review_count = storage_manager.count_lines(review_uri)
        else:
            staged_review_path, staged_business_path, input_location = stage_yelp_source_files(
                settings=settings,
                run_dir=run_dir,
            )
            review_count = count_lines(staged_review_path)

        manifest_payload = _build_manifest_payload(
            run_context=run_context,
            input_location=input_location,
            review_count=review_count,
        )
        write_json(manifest_path, manifest_payload)

        if settings.yelp_dataset_s3_uri:
            copy_source_object(
                source=SOURCE_NAME,
                source_uri=review_uri,
                destination_uri=run_prefix + REVIEW_FILENAME,
                storage_manager=storage_manager,
                run_context=run_context,
                logger=logger,
            )
            copy_source_object(
                source=SOURCE_NAME,
                source_uri=business_uri,
                destination_uri=run_prefix + BUSINESS_FILENAME,
                storage_manager=storage_manager,
                run_context=run_context,
                logger=logger,
            )
            upload_source_file(
                source=SOURCE_NAME,
                local_path=manifest_path,
                storage_manager=storage_manager,
                run_context=run_context,
                logger=logger,
                destination_uri=run_prefix + manifest_path.name,
            )
        else:
            for local_path in (staged_review_path, staged_business_path, manifest_path):
                upload_source_file(
                    source=SOURCE_NAME,
                    local_path=local_path,
                    storage_manager=storage_manager,
                    run_context=run_context,
                    logger=logger,
                    destination_uri=run_prefix + local_path.name,
                )

        run_prefix, current_prefix = promote_source_run(
            source=SOURCE_NAME,
            record_count=review_count,
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        log_event(
            logger,
            "source_fetch_completed",
            **run_context.as_log_fields(),
            input_path=input_location,
            output_path=current_prefix,
            record_count=review_count,
            file_count=3,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=current_prefix,
            record_count=review_count,
            file_count=3,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        return SourceIngestionResult(
            source=SOURCE_NAME,
            run_id=run_context.run_id,
            record_count=review_count,
            file_count=3,
            local_paths=tuple(
                path.resolve()
                for path in (staged_review_path, staged_business_path, manifest_path)
                if path is not None
            ),
            run_prefix=run_prefix,
            current_prefix=current_prefix,
        )
    except Exception as error:
        log_event(
            logger,
            "source_fetch_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            input_path=str(settings.yelp_dataset_source) if settings.yelp_dataset_source else None,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
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
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
 
 
def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    run()
 
 
if __name__ == "__main__":
    main()
 
