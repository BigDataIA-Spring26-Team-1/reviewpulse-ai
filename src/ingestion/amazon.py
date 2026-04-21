"""Amazon Reviews 2023 ingestion runner."""
 
from __future__ import annotations
 
import logging
import time
from typing import Any
 
from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import SourceIngestionResult, build_output_path, publish_source_files, write_jsonl
 
 
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
 
 
def fetch_amazon_records(settings: Settings) -> list[dict[str, Any]]:
    if settings.amazon_max_records <= 0:
        raise RuntimeError("AMAZON_MAX_RECORDS must be greater than zero.")
 
    load_dataset = _load_dataset_module()
    dataset_config = build_dataset_config(settings.amazon_category)
    dataset = load_dataset(
        settings.amazon_dataset_name,
        name=dataset_config,
        split="full",
        streaming=True,
        trust_remote_code=True,
        token=settings.huggingface_token or None,
    )
 
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if index >= settings.amazon_max_records:
            break
        records.append(dict(row))
 
    if not records:
        raise RuntimeError("Amazon ingestion returned zero records from Hugging Face streaming.")
    return records
 
 
def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_amazon", source="amazon")
    logger = logger or get_logger("ingestion.amazon")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
 
    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
 
    try:
        dataset_config = build_dataset_config(settings.amazon_category)
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
            status="started",
        )
        records = fetch_amazon_records(settings)
        output_path = build_output_path(settings, "amazon", "amazon_reviews.jsonl")
        write_jsonl(output_path, records)
        result = publish_source_files(
            source="amazon",
            local_paths=[output_path],
            record_count=len(records),
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        log_event(
            logger,
            "source_fetch_completed",
            **run_context.as_log_fields(),
            input_path=f"{settings.amazon_dataset_name}/{dataset_config}",
            output_path=str(output_path),
            record_count=len(records),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=str(output_path),
            record_count=len(records),
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
            input_path=f"{settings.amazon_dataset_name}/{build_dataset_config(settings.amazon_category)}",
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
 