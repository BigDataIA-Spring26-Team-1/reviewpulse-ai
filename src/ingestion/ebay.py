"""eBay Browse API ingestion runner."""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Iterator
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
    write_jsonl,
)


PRODUCTION_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
PRODUCTION_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
PRODUCTION_TAXONOMY_API_ROOT = "https://api.ebay.com/commerce/taxonomy/v1"
SANDBOX_TOKEN_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
SANDBOX_SEARCH_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
SANDBOX_TAXONOMY_API_ROOT = "https://api.sandbox.ebay.com/commerce/taxonomy/v1"
ALL_BUYING_OPTIONS_FILTER = "buyingOptions:{FIXED_PRICE|AUCTION|BEST_OFFER|CLASSIFIED_AD}"
SEARCH_MAX_ATTEMPTS = 3
RETRYABLE_SEARCH_STATUS_CODES = {429, 500, 502, 503, 504}


def resolve_ebay_api_urls(settings: Settings) -> tuple[str, str]:
    if settings.ebay_environment == "sandbox":
        return SANDBOX_TOKEN_URL, SANDBOX_SEARCH_URL
    return PRODUCTION_TOKEN_URL, PRODUCTION_SEARCH_URL


def resolve_ebay_taxonomy_api_root(settings: Settings) -> str:
    if settings.ebay_environment == "sandbox":
        return SANDBOX_TAXONOMY_API_ROOT
    return PRODUCTION_TAXONOMY_API_ROOT


def _browse_headers(access_token: str, settings: Settings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": settings.ebay_marketplace_id,
    }


def _taxonomy_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept-Encoding": "gzip",
    }


def _perform_search_request(
    *,
    session: requests.Session,
    search_url: str,
    access_token: str,
    settings: Settings,
    params: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, SEARCH_MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                search_url,
                headers=_browse_headers(access_token, settings),
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("eBay Browse search response was not a JSON object.")
            return payload
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
            if attempt >= SEARCH_MAX_ATTEMPTS:
                raise
            time.sleep(min(2 ** (attempt - 1), 4))
        except requests.HTTPError as error:
            last_error = error
            status_code = getattr(error.response, "status_code", None)
            if status_code not in RETRYABLE_SEARCH_STATUS_CODES or attempt >= SEARCH_MAX_ATTEMPTS:
                raise
            time.sleep(min(2 ** (attempt - 1), 4))

    if last_error is not None:
        raise last_error
    raise RuntimeError("eBay Browse search request failed without an explicit error.")


def get_access_token(settings: Settings, session: requests.Session | None = None) -> str:
    if not settings.ebay_app_id or not settings.ebay_cert_id:
        raise RuntimeError("EBAY_APP_ID and EBAY_CERT_ID must be configured for eBay ingestion.")

    active_session = session or requests.Session()
    token_url, _ = resolve_ebay_api_urls(settings)
    credentials = base64.b64encode(f"{settings.ebay_app_id}:{settings.ebay_cert_id}".encode("utf-8")).decode("utf-8")
    response = active_session.post(
        token_url,
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


def get_default_category_tree_id(
    *,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> str:
    active_session = session or requests.Session()
    taxonomy_root = resolve_ebay_taxonomy_api_root(settings)
    response = active_session.get(
        f"{taxonomy_root}/get_default_category_tree_id",
        headers=_taxonomy_headers(access_token),
        params={"marketplace_id": settings.ebay_marketplace_id},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    category_tree_id = str(payload.get("categoryTreeId", "")).strip()
    if not category_tree_id:
        raise RuntimeError("eBay taxonomy response did not include a categoryTreeId.")
    return category_tree_id


def get_category_tree(
    *,
    category_tree_id: str,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    active_session = session or requests.Session()
    taxonomy_root = resolve_ebay_taxonomy_api_root(settings)
    response = active_session.get(
        f"{taxonomy_root}/category_tree/{category_tree_id}",
        headers=_taxonomy_headers(access_token),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("eBay category tree response was not a JSON object.")
    return payload


def get_category_subtree(
    *,
    category_tree_id: str,
    category_id: str,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    active_session = session or requests.Session()
    taxonomy_root = resolve_ebay_taxonomy_api_root(settings)
    response = active_session.get(
        f"{taxonomy_root}/category_tree/{category_tree_id}/get_category_subtree",
        headers=_taxonomy_headers(access_token),
        params={"category_id": category_id},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("eBay category subtree response was not a JSON object.")
    return payload


def _is_leaf_category_node(node: dict[str, Any]) -> bool:
    children = node.get("childCategoryTreeNodes") or []
    return bool(node.get("leafCategoryTreeNode")) or not children


def _iter_leaf_categories(node: dict[str, Any], ancestors: tuple[str, ...] = ()) -> Iterator[dict[str, str]]:
    category = node.get("category") or {}
    category_id = str(category.get("categoryId") or "").strip()
    category_name = str(category.get("categoryName") or "").strip()
    current_path = ancestors + ((category_name,) if category_name else ())

    if _is_leaf_category_node(node):
        if category_id:
            yield {
                "category_id": category_id,
                "category_name": category_name or category_id,
                "category_path": " > ".join(part for part in current_path if part),
            }
        return

    for child in node.get("childCategoryTreeNodes") or []:
        if isinstance(child, dict):
            yield from _iter_leaf_categories(child, current_path)


def discover_leaf_categories(
    *,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> tuple[dict[str, str], ...]:
    if not settings.ebay_category_ids and not settings.ebay_crawl_all_categories:
        return tuple()

    active_session = session or requests.Session()
    category_tree_id = get_default_category_tree_id(
        access_token=access_token,
        settings=settings,
        session=active_session,
    )
    category_targets: dict[str, dict[str, str]] = {}

    if settings.ebay_category_ids:
        for category_id in settings.ebay_category_ids:
            payload = get_category_subtree(
                category_tree_id=category_tree_id,
                category_id=category_id,
                access_token=access_token,
                settings=settings,
                session=active_session,
            )
            subtree_node = payload.get("categorySubtreeNode")
            if not isinstance(subtree_node, dict):
                raise RuntimeError(f"eBay category subtree response was missing categorySubtreeNode for {category_id}.")
            for category in _iter_leaf_categories(subtree_node):
                category_targets.setdefault(category["category_id"], category)
    else:
        payload = get_category_tree(
            category_tree_id=category_tree_id,
            access_token=access_token,
            settings=settings,
            session=active_session,
        )
        root_node = payload.get("rootCategoryNode")
        if not isinstance(root_node, dict):
            raise RuntimeError("eBay category tree response was missing rootCategoryNode.")
        for category in _iter_leaf_categories(root_node):
            category_targets.setdefault(category["category_id"], category)

    return tuple(sorted(category_targets.values(), key=lambda item: (item["category_path"], item["category_id"])))


def _extract_category(item: dict[str, Any]) -> str:
    categories = item.get("categories")
    if isinstance(categories, list) and categories:
        first = categories[0]
        if isinstance(first, dict):
            return str(first.get("categoryName") or first.get("categoryId") or "unknown")
    return str(item.get("categoryPath") or "unknown")


def map_item_summary(
    item: dict[str, Any],
    query: str | None = None,
    *,
    crawl_mode: str = "query",
    crawl_category: dict[str, str] | None = None,
) -> dict[str, Any]:
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
        "search_query": query or "",
        "crawl_mode": crawl_mode,
        "crawl_category_id": crawl_category["category_id"] if crawl_category else None,
        "crawl_category_name": crawl_category["category_name"] if crawl_category else None,
        "crawl_category_path": crawl_category["category_path"] if crawl_category else None,
    }


def _fetch_search_items(
    *,
    params: dict[str, Any],
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
    query: str | None = None,
    crawl_mode: str,
    crawl_category: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    active_session = session or requests.Session()
    _, search_url = resolve_ebay_api_urls(settings)
    max_items = settings.ebay_max_items_per_query
    unlimited_results = max_items <= 0
    page_size = 200 if unlimited_results else min(max_items, 200)
    records: list[dict[str, Any]] = []
    offset = 0
    seen_ids: set[str] = set()

    while unlimited_results or len(records) < max_items:
        payload = _perform_search_request(
            session=active_session,
            search_url=search_url,
            access_token=access_token,
            settings=settings,
            params={
                **params,
                "filter": ALL_BUYING_OPTIONS_FILTER,
                "limit": page_size,
                "offset": offset,
            },
        )
        item_summaries = payload.get("itemSummaries") or []
        if not item_summaries:
            break

        for item in item_summaries:
            item_id = str(item.get("itemId") or "").strip()
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            records.append(
                map_item_summary(
                    item,
                    query,
                    crawl_mode=crawl_mode,
                    crawl_category=crawl_category,
                )
            )
            if not unlimited_results and len(records) >= max_items:
                break

        if not payload.get("next"):
            break
        offset += page_size

    return records


def fetch_query_items(
    query: str,
    *,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    return _fetch_search_items(
        params={"q": query},
        access_token=access_token,
        settings=settings,
        session=session,
        query=query,
        crawl_mode="query",
    )


def fetch_category_items(
    category: dict[str, str],
    *,
    access_token: str,
    settings: Settings,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    return _fetch_search_items(
        params={"category_ids": category["category_id"]},
        access_token=access_token,
        settings=settings,
        session=session,
        crawl_mode="category",
        crawl_category=category,
    )


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
        if not (settings.ebay_search_queries or settings.ebay_category_ids or settings.ebay_crawl_all_categories):
            raise RuntimeError(
                "Configure EBAY_SEARCH_QUERIES, EBAY_CATEGORY_IDS, or EBAY_CRAWL_ALL_CATEGORIES for eBay ingestion."
            )
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            status="started",
        )
        active_session = session or requests.Session()
        access_token = get_access_token(settings, active_session)
        output_path = build_output_path(settings, "ebay", "ebay_listings.jsonl")
        output_path.unlink(missing_ok=True)
        seen_item_ids: set[str] = set()
        record_count = 0

        def _persist_records(records: list[dict[str, Any]]) -> None:
            nonlocal record_count
            if not records:
                return
            if record_count == 0:
                write_jsonl(output_path, records)
            else:
                append_jsonl(output_path, records)
            record_count += len(records)

        if settings.ebay_category_ids or settings.ebay_crawl_all_categories:
            discovery_started_at = time.perf_counter()
            categories = discover_leaf_categories(
                access_token=access_token,
                settings=settings,
                session=active_session,
            )
            log_event(
                logger,
                "category_discovery_completed",
                **run_context.as_log_fields(),
                category_count=len(categories),
                duration_ms=round((time.perf_counter() - discovery_started_at) * 1000, 2),
                status="success",
            )
            if not categories:
                raise RuntimeError("eBay category crawl did not resolve any leaf categories.")

            for category in categories:
                category_started_at = time.perf_counter()
                try:
                    category_records = fetch_category_items(
                        category,
                        access_token=access_token,
                        settings=settings,
                        session=active_session,
                    )
                except Exception as error:
                    log_event(
                        logger,
                        "source_fetch_failed",
                        level=logging.ERROR,
                        **run_context.as_log_fields(),
                        input_path=f"{category['category_id']}::{category['category_path']}",
                        duration_ms=round((time.perf_counter() - category_started_at) * 1000, 2),
                        status="failed",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    continue
                new_records = [
                    record
                    for record in category_records
                    if str(record.get("item_id") or "").strip()
                    and str(record.get("item_id") or "").strip() not in seen_item_ids
                ]
                for record in new_records:
                    seen_item_ids.add(str(record["item_id"]).strip())
                _persist_records(new_records)
                log_event(
                    logger,
                    "source_fetch_completed",
                    **run_context.as_log_fields(),
                    input_path=f"{category['category_id']}::{category['category_path']}",
                    record_count=len(new_records),
                    duration_ms=round((time.perf_counter() - category_started_at) * 1000, 2),
                    status="success",
                )
        else:
            for query in settings.ebay_search_queries:
                query_started_at = time.perf_counter()
                try:
                    query_records = fetch_query_items(
                        query,
                        access_token=access_token,
                        settings=settings,
                        session=active_session,
                    )
                except Exception as error:
                    log_event(
                        logger,
                        "source_fetch_failed",
                        level=logging.ERROR,
                        **run_context.as_log_fields(),
                        input_path=query,
                        duration_ms=round((time.perf_counter() - query_started_at) * 1000, 2),
                        status="failed",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    continue
                new_records = [
                    record
                    for record in query_records
                    if str(record.get("item_id") or "").strip()
                    and str(record.get("item_id") or "").strip() not in seen_item_ids
                ]
                for record in new_records:
                    seen_item_ids.add(str(record["item_id"]).strip())
                _persist_records(new_records)
                log_event(
                    logger,
                    "source_fetch_completed",
                    **run_context.as_log_fields(),
                    input_path=query,
                    record_count=len(new_records),
                    duration_ms=round((time.perf_counter() - query_started_at) * 1000, 2),
                    status="success",
                )

        if record_count <= 0:
            raise RuntimeError("eBay ingestion returned zero items for the configured crawl settings.")

        result = publish_source_files(
            source="ebay",
            local_paths=[output_path],
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
