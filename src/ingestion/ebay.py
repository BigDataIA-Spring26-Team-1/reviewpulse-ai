"""eBay Browse API ingestion runner."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import requests

from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import SourceIngestionResult, build_output_path, publish_source_files, write_jsonl


TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


def get_access_token(settings: Settings, session: requests.Session | None = None) -> str:
    if not settings.ebay_app_id or not settings.ebay_cert_id:
        raise RuntimeError("EBAY_APP_ID and EBAY_CERT_ID must be configured for eBay ingestion.")

    active_session = session or requests.Session()
    credentials = base64.b64encode(f"{settings.ebay_app_id}:{settings.ebay_cert_id}".encode("utf-8")).decode("utf-8")
    response = active_session.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = str(payload.get("access_token", "")).strip()
    if not access_token:
        raise RuntimeError("eBay OAuth token response did not include an access_token.")
    return access_token


def _extract_category(item: dict[str, Any]) -> str:
    categories = item.get("categories")
    if isinstance(categories, list) and categories:
        first = categories[0]
        if isinstance(first, dict):
            return str(first.get("categoryName") or first.get("categoryId") or "unknown")
    return str(item.get("categoryPath") or "unknown")


def map_item_summary(item: dict[str, Any], query: str) -> dict[str, Any]:
    seller = item.get("seller") or {}
    feedback_percentage = seller.get("feedbackPercentage")
    if isinstance(feedback_percentage, str):
        feedback_percentage = feedback_percentage.replace("%", "").strip()

    category = _extract_category(item)
    return {
        "item_id": item.get("itemId"),
        "title": item.get("title"),
        "item_title": item.get("title"),
        "seller_rating": feedback_percentage,
        "feedback_count": seller.get("feedbackScore"),
        "feedback_text": item.get("shortDescription") or item.get("subtitle") or "",
        "category": category,
        "primary_category": category,
        "listing_date": item.get("itemCreationDate") or item.get("creationDate"),
        "seller_id": seller.get("username"),
        "url": item.get("itemWebUrl") or item.get("itemAffiliateWebUrl"),
        "condition": item.get("condition"),
        "price": (item.get("price") or {}).get("value"),
        "search_query": query,
    }


def fetch_query_items(
    query: str,
    *,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    if settings.ebay_max_items_per_query <= 0:
        raise RuntimeError("EBAY_MAX_ITEMS_PER_QUERY must be greater than zero.")

    active_session = session or requests.Session()
    page_size = min(settings.ebay_max_items_per_query, 200)
    records: list[dict[str, Any]] = []
    offset = 0
    seen_ids: set[str] = set()

    while len(records) < settings.ebay_max_items_per_query:
        response = active_session.get(
            SEARCH_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-EBAY-C-MARKETPLACE-ID": settings.ebay_marketplace_id,
            },
            params={
                "q": query,
                "limit": page_size,
                "offset": offset,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        item_summaries = payload.get("itemSummaries") or []
        if not item_summaries:
            break

        for item in item_summaries:
            item_id = str(item.get("itemId") or "").strip()
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            records.append(map_item_summary(item, query))
            if len(records) >= settings.ebay_max_items_per_query:
                break

        if not payload.get("next"):
            break
        offset += page_size

    return records


def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
    session: requests.Session | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_ebay", source="ebay")
    logger = logger or get_logger("ingestion.ebay")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()

    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")

    try:
        if not settings.ebay_search_queries:
            raise RuntimeError("EBAY_SEARCH_QUERIES must be configured for eBay ingestion.")
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            status="started",
        )
        active_session = session or requests.Session()
        access_token = get_access_token(settings, active_session)
        records_by_id: dict[str, dict[str, Any]] = {}

        for query in settings.ebay_search_queries:
            query_started_at = time.perf_counter()
            query_records = fetch_query_items(
                query,
                access_token=access_token,
                settings=settings,
                session=active_session,
            )
            for record in query_records:
                item_id = str(record.get("item_id") or "").strip()
                if item_id and item_id not in records_by_id:
                    records_by_id[item_id] = record
            log_event(
                logger,
                "source_fetch_completed",
                **run_context.as_log_fields(),
                input_path=query,
                record_count=len(query_records),
                duration_ms=round((time.perf_counter() - query_started_at) * 1000, 2),
                status="success",
            )

        records = list(records_by_id.values())
        if not records:
            raise RuntimeError("eBay ingestion returned zero items for the configured queries.")

        output_path = build_output_path(settings, "ebay", "ebay_listings.jsonl")
        write_jsonl(output_path, records)
        result = publish_source_files(
            source="ebay",
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
