"""iFixit guide ingestion runner."""
 
from __future__ import annotations
 
import logging
import time
from typing import Any
 
import requests
 
from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import SourceIngestionResult, build_output_path, publish_source_files, write_jsonl
 
 
def _pick(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None
 
 
def fetch_guide_payload(
    guide_id: str,
    *,
    base_url: str,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    active_session = session or requests.Session()
    response = active_session.get(
        f"{base_url.rstrip('/')}/api/2.0/guides/{guide_id}",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected iFixit API payload for guide {guide_id}.")
    return payload
 
 
def map_guide_payload(payload: dict[str, Any], *, guide_id: str, base_url: str) -> dict[str, Any]:
    author = payload.get("author")
    if isinstance(author, dict):
        author_name = str(author.get("username") or author.get("name") or "unknown")
    else:
        author_name = str(author or "unknown")
 
    category = _pick(payload, "category", "category_title", "category_name") or "repairability"
    if isinstance(category, dict):
        category = category.get("title") or category.get("name") or "repairability"
 
    return {
        "guide_id": _pick(payload, "guideid", "guide_id") or guide_id,
        "title": _pick(payload, "title", "subject") or guide_id,
        "device_name": _pick(payload, "subject", "device_name", "device", "title") or guide_id,
        "device_category": str(category),
        "repairability_score": _pick(payload, "repairability_score", "repairabilityScore"),
        "review_text": _pick(
            payload,
            "summary",
            "introduction_raw",
            "introduction_rendered",
            "introduction",
            "conclusion",
        )
        or "",
        "published_date": _pick(payload, "published_date", "created_date", "createdAt", "created_at"),
        "author": author_name,
        "helpful_votes": _pick(payload, "likes", "likes_count", "favorites", "favorites_count"),
        "url": _pick(payload, "url") or f"{base_url.rstrip('/')}/Guide/{guide_id}",
    }
 
 
def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
    session: requests.Session | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_ifixit", source="ifixit")
    logger = logger or get_logger("ingestion.ifixit")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
 
    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
 
    try:
        if not settings.ifixit_guide_ids:
            raise RuntimeError("IFIXIT_GUIDE_IDS must be configured for iFixit ingestion.")
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=settings.ifixit_base_url,
            status="started",
        )
        active_session = session or requests.Session()
        records: list[dict[str, Any]] = []
        for guide_id in settings.ifixit_guide_ids:
            guide_started_at = time.perf_counter()
            payload = fetch_guide_payload(
                guide_id,
                base_url=settings.ifixit_base_url,
                session=active_session,
            )
            records.append(map_guide_payload(payload, guide_id=guide_id, base_url=settings.ifixit_base_url))
            log_event(
                logger,
                "source_fetch_completed",
                **run_context.as_log_fields(),
                input_path=guide_id,
                record_count=1,
                duration_ms=round((time.perf_counter() - guide_started_at) * 1000, 2),
                status="success",
            )
 
        if not records:
            raise RuntimeError("iFixit ingestion returned zero guide records.")
 
        output_path = build_output_path(settings, "ifixit", "ifixit_guides.jsonl")
        write_jsonl(output_path, records)
        result = publish_source_files(
            source="ifixit",
            local_paths=[output_path],
            record_count=len(records),
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
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
            input_path=settings.ifixit_base_url,
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
 
 