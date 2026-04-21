"""Yelp Open Dataset ingestion runner."""
 
from __future__ import annotations
 
import logging
import time
from pathlib import Path
 
from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import (
    SourceIngestionResult,
    build_output_path,
    copy_file,
    count_lines,
    publish_source_files,
)
 
 
REVIEW_FILENAME = "yelp_academic_dataset_review.json"
BUSINESS_FILENAME = "yelp_academic_dataset_business.json"
 
 
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
 
 
def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_yelp", source="yelp")
    logger = logger or get_logger("ingestion.yelp")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
 
    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
 
    try:
        if settings.yelp_dataset_path is None:
            raise RuntimeError("YELP_DATASET_PATH must be configured for Yelp ingestion.")
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=str(settings.yelp_dataset_path),
            status="started",
        )
        review_path, business_path = resolve_yelp_source_files(settings.yelp_dataset_path)
        local_review_path = build_output_path(settings, "yelp", REVIEW_FILENAME)
        local_business_path = build_output_path(settings, "yelp", BUSINESS_FILENAME)
        copy_file(review_path, local_review_path)
        copy_file(business_path, local_business_path)
        review_count = count_lines(local_review_path)
        result = publish_source_files(
            source="yelp",
            local_paths=[local_review_path, local_business_path],
            record_count=review_count,
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        log_event(
            logger,
            "source_fetch_completed",
            **run_context.as_log_fields(),
            input_path=str(settings.yelp_dataset_path),
            output_path=str(local_review_path.parent),
            record_count=review_count,
            file_count=2,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=str(local_review_path.parent),
            record_count=review_count,
            file_count=2,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        return result
    except Exception as error:
        log_event(
            logger,
            "source_fetch_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            input_path=str(settings.yelp_dataset_path) if settings.yelp_dataset_path else None,
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
 