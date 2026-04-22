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
from src.ingestion.common import (
    SourceIngestionResult,
    append_jsonl,
    build_output_path,
    publish_source_files,
)


WRITE_BATCH_SIZE = 100
PROGRESS_LOG_INTERVAL = 100
 
 
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


def fetch_public_guides_page(
    *,
    base_url: str,
    offset: int,
    limit: int,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    active_session = session or requests.Session()
    response = active_session.get(
        f"{base_url.rstrip('/')}/api/2.0/guides",
        params={
            "order": "DESC",
            "offset": offset,
            "limit": limit,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected iFixit guides payload while listing public guides.")
    return [guide for guide in payload if isinstance(guide, dict)]


def resolve_guide_ids(
    settings: Settings,
    *,
    session: requests.Session | None = None,
) -> tuple[str, ...]:
    if settings.ifixit_guide_ids:
        return settings.ifixit_guide_ids
    if settings.ifixit_max_guides <= 0:
        raise RuntimeError("Configure IFIXIT_GUIDE_IDS or set IFIXIT_MAX_GUIDES to ingest iFixit guides.")

    discovered_ids: list[str] = []
    seen_ids: set[str] = set()
    offset = 0
    page_size = 200

    while len(discovered_ids) < settings.ifixit_max_guides:
        remaining = settings.ifixit_max_guides - len(discovered_ids)
        guides = fetch_public_guides_page(
            base_url=settings.ifixit_base_url,
            offset=offset,
            limit=min(page_size, remaining),
            session=session,
        )
        if not guides:
            break

        page_added = 0
        for guide in guides:
            guide_id = _pick(guide, "guideid", "guide_id", "id")
            if guide_id in (None, ""):
                continue
            guide_id_value = str(guide_id)
            if guide_id_value in seen_ids:
                continue
            seen_ids.add(guide_id_value)
            discovered_ids.append(guide_id_value)
            page_added += 1
            if len(discovered_ids) >= settings.ifixit_max_guides:
                break

        if page_added == 0 or len(guides) < min(page_size, remaining):
            break
        offset += len(guides)

    if not discovered_ids:
        raise RuntimeError("iFixit public guide discovery returned zero guide IDs.")
    return tuple(discovered_ids)


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
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            input_path=settings.ifixit_base_url,
            status="started",
        )
        active_session = session or requests.Session()
        guide_ids = resolve_guide_ids(settings, session=active_session)
        log_event(
            logger,
            "guide_discovery_completed",
            **run_context.as_log_fields(),
            input_path=settings.ifixit_base_url,
            record_count=len(guide_ids),
            status="success",
        )
        output_path = build_output_path(settings, "ifixit", "ifixit_guides.jsonl")
        raw_output_path = build_output_path(settings, "ifixit", "ifixit_guides_raw.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        raw_output_path.write_text("", encoding="utf-8")

        records_batch: list[dict[str, Any]] = []
        raw_records_batch: list[dict[str, Any]] = []
        record_count = 0
        total_guides = len(guide_ids)

        for index, guide_id in enumerate(guide_ids, start=1):
            guide_started_at = time.perf_counter()
            try:
                payload = fetch_guide_payload(
                    guide_id,
                    base_url=settings.ifixit_base_url,
                    session=active_session,
                )
            except Exception as error:
                log_event(
                    logger,
                    "source_fetch_failed",
                    level=logging.ERROR,
                    **run_context.as_log_fields(),
                    input_path=guide_id,
                    duration_ms=round((time.perf_counter() - guide_started_at) * 1000, 2),
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                continue

            raw_records_batch.append(payload)
            records_batch.append(map_guide_payload(payload, guide_id=guide_id, base_url=settings.ifixit_base_url))
            record_count += 1

            if len(records_batch) >= WRITE_BATCH_SIZE:
                append_jsonl(output_path, records_batch)
                append_jsonl(raw_output_path, raw_records_batch)
                records_batch.clear()
                raw_records_batch.clear()

            if index <= 5 or index == total_guides or index % PROGRESS_LOG_INTERVAL == 0:
                log_event(
                    logger,
                    "source_fetch_completed",
                    **run_context.as_log_fields(),
                    input_path=guide_id,
                    record_count=1,
                    completed_guides=index,
                    total_guides=total_guides,
                    cumulative_records=record_count,
                    duration_ms=round((time.perf_counter() - guide_started_at) * 1000, 2),
                    status="success",
                )

        if records_batch:
            append_jsonl(output_path, records_batch)
            append_jsonl(raw_output_path, raw_records_batch)

        if record_count <= 0:
            raise RuntimeError("iFixit ingestion returned zero guide records.")
        result = publish_source_files(
            source="ifixit",
            local_paths=[output_path, raw_output_path],
            record_count=record_count,
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=str(output_path),
            record_count=record_count,
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
 
 
