"""Unified schema and source normalization logic."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event


PROPOSAL_CORE_SOURCES = ("amazon", "yelp", "ebay", "ifixit", "youtube")
OPTIONAL_SOURCES = ("reddit",)

AMAZON_FILE_CANDIDATES = (
    "amazon/current",
    "amazon_electronics_sample.jsonl",
    "amazon/amazon_electronics_sample.jsonl",
    "amazon/amazon_reviews.jsonl",
)
YELP_REVIEW_FILE_CANDIDATES = (
    "yelp/current/yelp_academic_dataset_review.json",
    "yelp/yelp_academic_dataset_review.json",
    "yelp/yelp_reviews.jsonl",
    "yelp_reviews.jsonl",
)
YELP_BUSINESS_FILE_CANDIDATES = (
    "yelp/current/yelp_academic_dataset_business.json",
    "yelp/yelp_academic_dataset_business.json",
    "yelp/yelp_businesses.jsonl",
    "yelp_business.jsonl",
)
EBAY_FILE_CANDIDATES = (
    "ebay_reviews.jsonl",
    "ebay/current/ebay_reviews.jsonl",
    "ebay/ebay_reviews.jsonl",
    "ebay/current/ebay_listings.jsonl",
    "ebay/ebay_listings.jsonl",
    "ebay/current/listings.jsonl",
    "ebay/listings.jsonl",
)
IFIXIT_FILE_CANDIDATES = (
    "ifixit_reviews.jsonl",
    "ifixit/current/ifixit_reviews.jsonl",
    "ifixit/ifixit_reviews.jsonl",
    "ifixit/current/ifixit_guides.jsonl",
    "ifixit/ifixit_guides.jsonl",
    "ifixit/current/guides.jsonl",
    "ifixit/guides.jsonl",
)
YOUTUBE_FILE_CANDIDATES = (
    "youtube_reviews.jsonl",
    "youtube/current/youtube_reviews.jsonl",
    "youtube/youtube_reviews.jsonl",
    "youtube/current/youtube_transcripts.jsonl",
    "youtube/youtube_transcripts.jsonl",
)
REDDIT_FILE_CANDIDATES = (
    "reddit_reviews.jsonl",
    "reddit/current/reddit_reviews.jsonl",
    "reddit/reddit_reviews.jsonl",
)

UNIFIED_REVIEW_FIELDS = (
    "review_id",
    "product_name",
    "product_category",
    "source",
    "rating_normalized",
    "review_text",
    "review_date",
    "reviewer_id",
    "verified_purchase",
    "helpful_votes",
    "source_url",
    "display_name",
    "display_category",
    "entity_type",
    "text_length_words",
)


@dataclass(frozen=True, slots=True)
class UnifiedReviewRecord:
    review_id: str
    product_name: str
    product_category: str
    source: str
    rating_normalized: Optional[float]
    review_text: str
    review_date: Optional[str]
    reviewer_id: str
    verified_purchase: Optional[bool]
    helpful_votes: Optional[int]
    source_url: str
    display_name: str
    display_category: str
    entity_type: str
    text_length_words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_text(*parts: Optional[str]) -> str:
    cleaned_parts = [str(part).strip() for part in parts if part and str(part).strip()]
    return ". ".join(cleaned_parts)


def _word_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len([token for token in str(text).split() if token])


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_unit_scale(value: Any, minimum: float, maximum: float) -> Optional[float]:
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return None
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    clamped = _clamp(numeric_value, minimum, maximum)
    return round((clamped - minimum) / (maximum - minimum), 4)


def isoformat_from_unix_millis(value: Any) -> Optional[str]:
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return None
    try:
        return datetime.fromtimestamp(numeric_value / 1000, UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def isoformat_from_unix_seconds(value: Any) -> Optional[str]:
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return None
    try:
        return datetime.fromtimestamp(numeric_value, UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def isoformat_from_datetime_string(value: Any, append_midnight: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if append_midnight and len(text) == 10:
        return f"{text}T00:00:00"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def create_unified_record(
    review_id: str,
    product_name: str,
    product_category: str,
    source: str,
    rating_normalized: Optional[float],
    review_text: str,
    review_date: Optional[str],
    reviewer_id: str,
    verified_purchase: Optional[bool],
    helpful_votes: Optional[int],
    source_url: str,
    display_name: Optional[str] = None,
    display_category: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> dict[str, Any]:
    text = str(review_text or "")
    record = UnifiedReviewRecord(
        review_id=str(review_id),
        product_name=str(product_name or "unknown"),
        product_category=str(product_category or "unknown"),
        source=str(source),
        rating_normalized=rating_normalized,
        review_text=text,
        review_date=review_date,
        reviewer_id=str(reviewer_id or "unknown"),
        verified_purchase=verified_purchase,
        helpful_votes=helpful_votes,
        source_url=str(source_url or ""),
        display_name=str(display_name or product_name or "unknown"),
        display_category=str(display_category or product_category or "unknown"),
        entity_type=str(entity_type or f"{source}_entry"),
        text_length_words=_word_count(text),
    )
    return record.to_dict()


def normalize_amazon(raw: Mapping[str, Any]) -> dict[str, Any]:
    asin = str(raw.get("asin") or "unknown")
    parent_asin = str(raw.get("parent_asin") or asin)

    return create_unified_record(
        review_id=f"amazon_{asin}_{str(raw.get('user_id') or 'unknown')[:8]}",
        product_name=parent_asin,
        product_category=str(raw.get("product_category") or "electronics"),
        source="amazon",
        rating_normalized=normalize_unit_scale(raw.get("rating"), 1.0, 5.0),
        review_text=_clean_text(raw.get("title"), raw.get("text")),
        review_date=isoformat_from_unix_millis(raw.get("timestamp")),
        reviewer_id=str(raw.get("user_id") or "unknown"),
        verified_purchase=raw.get("verified_purchase"),
        helpful_votes=_coerce_int(raw.get("helpful_vote")),
        source_url=str(raw.get("url") or f"https://amazon.com/dp/{asin}"),
        display_name=str(raw.get("display_name") or raw.get("title") or f"Amazon Electronics Item {asin}"),
        display_category=str(raw.get("display_category") or "Electronics Product"),
        entity_type="product_review",
    )


def normalize_yelp(
    raw: Mapping[str, Any],
    business_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    business = {}
    if business_lookup:
        business = dict(business_lookup.get(str(raw.get("business_id")), {}))

    categories = business.get("categories") or raw.get("categories") or "local_business"
    display_category = "Local Business"
    if isinstance(categories, str) and categories.strip():
        display_category = categories.split(",")[0].strip()

    business_name = (
        business.get("name")
        or raw.get("business_name")
        or raw.get("business_id")
        or "unknown"
    )

    return create_unified_record(
        review_id=f"yelp_{raw.get('review_id') or 'unknown'}",
        product_name=str(business_name),
        product_category=str(categories),
        source="yelp",
        rating_normalized=normalize_unit_scale(raw.get("stars"), 1.0, 5.0),
        review_text=str(raw.get("text") or ""),
        review_date=isoformat_from_datetime_string(raw.get("date"), append_midnight=True),
        reviewer_id=str(raw.get("user_id") or "unknown"),
        verified_purchase=None,
        helpful_votes=None,
        source_url=str(raw.get("url") or f"https://yelp.com/biz/{raw.get('business_id') or ''}"),
        display_name=str(business_name),
        display_category=display_category,
        entity_type="business_review",
    )


def normalize_ebay(raw: Mapping[str, Any]) -> dict[str, Any]:
    item_id = str(raw.get("item_id") or "unknown")
    title = str(raw.get("title") or raw.get("item_title") or item_id)
    category = str(raw.get("category") or raw.get("primary_category") or "unknown")

    return create_unified_record(
        review_id=f"ebay_{item_id}",
        product_name=title,
        product_category=category,
        source="ebay",
        rating_normalized=normalize_unit_scale(raw.get("seller_rating"), 0.0, 100.0),
        review_text=str(raw.get("feedback_text") or raw.get("review_text") or ""),
        review_date=isoformat_from_datetime_string(raw.get("listing_date") or raw.get("review_date")),
        reviewer_id=str(raw.get("seller_id") or raw.get("seller") or raw.get("reviewer_id") or "unknown"),
        verified_purchase=None,
        helpful_votes=_coerce_int(raw.get("feedback_count") or raw.get("helpful_votes")),
        source_url=str(raw.get("url") or f"https://www.ebay.com/itm/{item_id}"),
        display_name=title,
        display_category=category,
        entity_type="listing_review",
    )




def normalize_ifixit(raw: Mapping[str, Any]) -> dict[str, Any]:
    guide_id = str(raw.get("guide_id") or raw.get("source_id") or "unknown")
    title = str(raw.get("title") or raw.get("device_name") or guide_id)
    device_name = str(raw.get("device_name") or title)
    category = str(raw.get("device_category") or "repairability")

    return create_unified_record(
        review_id=f"ifixit_{guide_id}",
        product_name=device_name,
        product_category=category,
        source="ifixit",
        rating_normalized=normalize_unit_scale(raw.get("repairability_score"), 1.0, 10.0),
        review_text=str(raw.get("review_text") or raw.get("text") or ""),
        review_date=(
            isoformat_from_datetime_string(raw.get("published_date"))
            or isoformat_from_unix_seconds(raw.get("published_date"))
        ),
        reviewer_id=str(raw.get("author") or "unknown"),
        verified_purchase=None,
        helpful_votes=_coerce_int(raw.get("helpful_votes")),
        source_url=str(raw.get("url") or f"https://www.ifixit.com/Guide/{guide_id}"),
        display_name=title,
        display_category=category,
        entity_type="repair_review",
    )


def normalize_reddit(raw: Mapping[str, Any]) -> dict[str, Any]:
    subreddit = str(raw.get("subreddit") or "general")
    title = str(raw.get("title") or "Reddit Discussion")

    return create_unified_record(
        review_id=f"reddit_{raw.get('source_id') or 'unknown'}",
        product_name=str(raw.get("product_name") or "unknown"),
        product_category=subreddit,
        source="reddit",
        rating_normalized=None,
        review_text=_clean_text(raw.get("title"), raw.get("text")),
        review_date=isoformat_from_unix_seconds(raw.get("created_utc")),
        reviewer_id=str(raw.get("author") or "unknown"),
        verified_purchase=None,
        helpful_votes=_coerce_int(raw.get("score")),
        source_url=str(raw.get("url") or ""),
        display_name=title,
        display_category=subreddit,
        entity_type="forum_post",
    )


def normalize_youtube(raw: Mapping[str, Any]) -> dict[str, Any]:
    title = str(raw.get("title") or "YouTube Review")

    return create_unified_record(
        review_id=f"youtube_{raw.get('source_id') or raw.get('video_id') or 'unknown'}",
        product_name=str(raw.get("product_name") or "unknown"),
        product_category=str(raw.get("product_category") or "unknown"),
        source="youtube",
        rating_normalized=None,
        review_text=str(raw.get("text") or raw.get("transcript") or ""),
        review_date=isoformat_from_unix_seconds(raw.get("created_utc"))
        or isoformat_from_datetime_string(raw.get("published_date")),
        reviewer_id=str(raw.get("channel") or raw.get("channel_id") or "unknown"),
        verified_purchase=None,
        helpful_votes=_coerce_int(raw.get("like_count")),
        source_url=str(raw.get("url") or ""),
        display_name=title,
        display_category=str(raw.get("display_category") or "Video Review"),
        entity_type="video_transcript",
    )


def load_jsonl(path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_paths = [path]
    if path.is_dir():
        input_paths = sorted(
            candidate.resolve()
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".jsonl"
        )

    loaded = 0
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and loaded >= limit:
                    return records
                payload = line.strip()
                if not payload:
                    continue
                try:
                    records.append(json.loads(payload))
                    loaded += 1
                except json.JSONDecodeError:
                    continue
    return records


def load_yelp_business_lookup(path: Path, limit: Optional[int] = None) -> dict[str, dict[str, Any]]:
    business_lookup: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path, limit=limit):
        business_id = row.get("business_id")
        if business_id:
            business_lookup[str(business_id)] = row
    return business_lookup


def find_first_existing_path(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = (base_dir / candidate).resolve()
        if path.exists():
            return path
    return None


def _directory_has_jsonl_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(candidate.is_file() and candidate.suffix.lower() == ".jsonl" for candidate in path.iterdir())


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_amazon_source_path(base_dir: Path) -> Optional[Path]:
    runs_dir = (base_dir / "amazon" / "runs").resolve()
    latest_completed: tuple[float, Path] | None = None
    if runs_dir.exists() and runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not any(run_dir.glob("amazon_reviews_batch_*.jsonl")):
                continue

            checkpoint_payload = _load_json_file(run_dir / "_checkpoint.json") or {}
            manifest_exists = (run_dir / "manifest.json").exists()
            if not manifest_exists and not checkpoint_payload.get("completed"):
                continue

            sort_key = max(
                run_dir.stat().st_mtime,
                (run_dir / "_checkpoint.json").stat().st_mtime if (run_dir / "_checkpoint.json").exists() else 0.0,
                (run_dir / "manifest.json").stat().st_mtime if (run_dir / "manifest.json").exists() else 0.0,
            )
            candidate = (sort_key, run_dir.resolve())
            if latest_completed is None or candidate[0] > latest_completed[0]:
                latest_completed = candidate

    if latest_completed is not None:
        return latest_completed[1]

    direct_path = find_first_existing_path(base_dir, AMAZON_FILE_CANDIDATES)
    if direct_path is not None:
        if direct_path.is_dir():
            if _directory_has_jsonl_files(direct_path):
                return direct_path
        else:
            return direct_path

    return None


def resolve_yelp_source_paths(base_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    runs_dir = (base_dir / "yelp" / "runs").resolve()
    latest_completed: tuple[float, Path, Path] | None = None
    if runs_dir.exists() and runs_dir.is_dir():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            review_path = (run_dir / "yelp_academic_dataset_review.json").resolve()
            business_path = (run_dir / "yelp_academic_dataset_business.json").resolve()
            manifest_path = (run_dir / "manifest.json").resolve()
            if not (review_path.exists() and business_path.exists() and manifest_path.exists()):
                continue

            sort_key = max(
                run_dir.stat().st_mtime,
                review_path.stat().st_mtime,
                business_path.stat().st_mtime,
                manifest_path.stat().st_mtime,
            )
            candidate = (sort_key, review_path, business_path)
            if latest_completed is None or candidate[0] > latest_completed[0]:
                latest_completed = candidate

    if latest_completed is not None:
        return latest_completed[1], latest_completed[2]

    paired_candidates = (
        (
            "yelp/current/yelp_academic_dataset_review.json",
            "yelp/current/yelp_academic_dataset_business.json",
        ),
        (
            "yelp/yelp_academic_dataset_review.json",
            "yelp/yelp_academic_dataset_business.json",
        ),
        (
            "yelp/yelp_reviews.jsonl",
            "yelp/yelp_businesses.jsonl",
        ),
        (
            "yelp_reviews.jsonl",
            "yelp_business.jsonl",
        ),
    )

    for review_candidate, business_candidate in paired_candidates:
        review_path = (base_dir / review_candidate).resolve()
        business_path = (base_dir / business_candidate).resolve()
        if review_path.exists() and business_path.exists():
            return review_path, business_path

    settings = get_settings()
    configured_dataset_path = settings.yelp_dataset_path
    if configured_dataset_path is not None:
        resolved = configured_dataset_path.resolve()
        if resolved.exists():
            if resolved.is_dir():
                review_filename = "yelp_academic_dataset_review.json"
                business_filename = "yelp_academic_dataset_business.json"
                review_path = (resolved / review_filename).resolve() if (resolved / review_filename).exists() else None
                business_path = (
                    (resolved / business_filename).resolve()
                    if (resolved / business_filename).exists()
                    else None
                )
            elif resolved.name == "yelp_academic_dataset_review.json":
                review_path = resolved
                business_candidate = (resolved.parent / "yelp_academic_dataset_business.json").resolve()
                business_path = business_candidate if business_candidate.exists() else None
            elif resolved.name == "yelp_academic_dataset_business.json":
                business_path = resolved
                review_candidate = (resolved.parent / "yelp_academic_dataset_review.json").resolve()
                review_path = review_candidate if review_candidate.exists() else None
            else:
                review_path = None
                business_path = None

            if review_path is not None and business_path is not None:
                return review_path, business_path

    return (
        find_first_existing_path(base_dir, YELP_REVIEW_FILE_CANDIDATES),
        find_first_existing_path(base_dir, YELP_BUSINESS_FILE_CANDIDATES),
    )


def resolve_source_input_path(base_dir: Path, source: str, candidates: Sequence[str]) -> Optional[Path]:
    if source == "amazon":
        return resolve_amazon_source_path(base_dir)
    if source == "yelp":
        review_path, _ = resolve_yelp_source_paths(base_dir)
        return review_path
    return find_first_existing_path(base_dir, candidates)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stage_s3_raw_current_sources(
    *,
    settings: Any | None = None,
    storage_manager: S3StorageManager | None = None,
    logger: logging.Logger | None = None,
    run_context: PipelineRunContext | None = None,
    sources: Sequence[str] = (*PROPOSAL_CORE_SOURCES, *OPTIONAL_SOURCES),
) -> dict[str, list[Path]]:
    resolved_settings = settings or get_settings()
    if not resolved_settings.s3_enabled:
        return {}
    should_stage = os.getenv("REVIEWPULSE_STAGE_S3_RAW", "1").strip().lower()
    if should_stage in {"0", "false", "no", "off"}:
        return {}

    resolved_storage_manager = storage_manager or S3StorageManager.from_settings(resolved_settings)
    resolved_logger = logger or get_logger("normalization.s3_staging")
    resolved_run_context = run_context or build_run_context(stage="stage_s3_raw_current")

    staged: dict[str, list[Path]] = {}
    for source_name in sources:
        source_prefix = resolved_storage_manager.resolver.raw_current_prefix(source_name)
        local_dir = (resolved_settings.data_dir / source_name / "current").resolve()
        log_event(
            resolved_logger,
            "s3_raw_stage_started",
            source=source_name,
            stage=resolved_run_context.stage,
            run_id=resolved_run_context.run_id,
            dag_id=resolved_run_context.dag_id,
            task_id=resolved_run_context.task_id,
            input_path=source_prefix,
            output_path=str(local_dir),
            status="started",
        )
        downloaded_paths = resolved_storage_manager.download_prefix(
            source_prefix,
            local_dir,
            clear_destination=True,
            exclude_latest_marker=True,
        )
        if not downloaded_paths:
            log_event(
                resolved_logger,
                "s3_raw_stage_skipped",
                source=source_name,
                stage=resolved_run_context.stage,
                run_id=resolved_run_context.run_id,
                dag_id=resolved_run_context.dag_id,
                task_id=resolved_run_context.task_id,
                input_path=source_prefix,
                output_path=str(local_dir),
                file_count=0,
                status="skipped",
            )
            continue

        staged[source_name] = downloaded_paths
        log_event(
            resolved_logger,
            "s3_raw_stage_completed",
            source=source_name,
            stage=resolved_run_context.stage,
            run_id=resolved_run_context.run_id,
            dag_id=resolved_run_context.dag_id,
            task_id=resolved_run_context.task_id,
            input_path=source_prefix,
            output_path=str(local_dir),
            file_count=len(downloaded_paths),
            status="success",
        )

    return staged


def _upload_raw_snapshot(
    storage_manager: S3StorageManager,
    logger: logging.Logger,
    run_context: PipelineRunContext,
    source_name: str,
    source_path: Path,
) -> str:
    destination_uri = storage_manager.resolver.raw_run_prefix(source_name, run_context.run_id) + source_path.name
    log_event(
        logger,
        "s3_upload_started",
        source=source_name,
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(source_path),
        output_path=destination_uri,
        status="started",
    )
    storage_manager.upload_file(source_path, destination_uri)
    log_event(
        logger,
        "s3_upload_completed",
        source=source_name,
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(source_path),
        output_path=destination_uri,
        file_count=1,
        status="success",
    )
    return destination_uri


def _publish_normalized_output(
    storage_manager: S3StorageManager,
    logger: logging.Logger,
    run_context: PipelineRunContext,
    local_output_path: Path,
) -> dict[str, Any]:
    run_prefix = storage_manager.resolver.processed_run_prefix("normalized_jsonl", run_context.run_id)
    current_prefix = storage_manager.resolver.processed_current_prefix("normalized_jsonl")
    destination_uri = run_prefix + local_output_path.name

    log_event(
        logger,
        "s3_upload_started",
        stage="normalized_jsonl",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(local_output_path),
        output_path=destination_uri,
        status="started",
    )
    storage_manager.upload_file(local_output_path, destination_uri)
    log_event(
        logger,
        "s3_upload_completed",
        stage="normalized_jsonl",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(local_output_path),
        output_path=destination_uri,
        file_count=1,
        status="success",
    )
    promotion = storage_manager.promote_run_prefix(
        run_prefix,
        current_prefix,
        run_id=run_context.run_id,
        metadata={"stage": "normalized_jsonl", "status": "success"},
    )
    log_event(
        logger,
        "latest_run_promoted",
        stage="normalized_jsonl",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        output_path=current_prefix,
        file_count=promotion["copied_count"],
        status="success",
    )
    return promotion


def run_local_normalization(
    include_optional_sources: bool = True,
    *,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
    publish_raw_snapshots: bool = False,
) -> list[dict[str, Any]]:
    settings = get_settings()
    run_context = run_context or build_run_context(stage="normalize_local_preview")
    logger = logger or get_logger("normalization.local")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)

    normalized_records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    available_sources: list[tuple[str, Path]] = []
    raw_sources_for_promotion: list[str] = []

    log_event(
        logger,
        "normalization_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        status="started",
    )

    stage_s3_raw_current_sources(
        settings=settings,
        storage_manager=storage_manager,
        logger=logger,
        run_context=run_context,
        sources=(
            (*PROPOSAL_CORE_SOURCES, *OPTIONAL_SOURCES)
            if include_optional_sources
            else PROPOSAL_CORE_SOURCES
        ),
    )

    amazon_path = resolve_source_input_path(settings.data_dir, "amazon", AMAZON_FILE_CANDIDATES)
    if amazon_path:
        source_started_at = time.perf_counter()
        log_event(
            logger,
            "source_fetch_started",
            source="amazon",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(amazon_path),
            status="started",
        )
        amazon_records = [normalize_amazon(row) for row in load_jsonl(amazon_path)]
        normalized_records.extend(amazon_records)
        source_counts["amazon"] = len(amazon_records)
        available_sources.append(("amazon", amazon_path))
        log_event(
            logger,
            "source_fetch_completed",
            source="amazon",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(amazon_path),
            record_count=len(amazon_records),
            duration_ms=round((time.perf_counter() - source_started_at) * 1000, 2),
            status="success",
        )

    yelp_review_path, yelp_business_path = resolve_yelp_source_paths(settings.data_dir)
    if yelp_review_path:
        source_started_at = time.perf_counter()
        log_event(
            logger,
            "source_fetch_started",
            source="yelp",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(yelp_review_path),
            status="started",
        )
        business_lookup = load_yelp_business_lookup(yelp_business_path) if yelp_business_path else {}
        yelp_records = [
            normalize_yelp(row, business_lookup)
            for row in load_jsonl(yelp_review_path)
        ]
        normalized_records.extend(yelp_records)
        source_counts["yelp"] = len(yelp_records)
        available_sources.append(("yelp", yelp_review_path))
        log_event(
            logger,
            "source_fetch_completed",
            source="yelp",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(yelp_review_path),
            record_count=len(yelp_records),
            duration_ms=round((time.perf_counter() - source_started_at) * 1000, 2),
            status="success",
        )

    ebay_path = find_first_existing_path(settings.data_dir, EBAY_FILE_CANDIDATES)
    if ebay_path:
        source_started_at = time.perf_counter()
        log_event(
            logger,
            "source_fetch_started",
            source="ebay",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(ebay_path),
            status="started",
        )
        ebay_records = [normalize_ebay(row) for row in load_jsonl(ebay_path)]
        normalized_records.extend(ebay_records)
        source_counts["ebay"] = len(ebay_records)
        available_sources.append(("ebay", ebay_path))
        log_event(
            logger,
            "source_fetch_completed",
            source="ebay",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(ebay_path),
            record_count=len(ebay_records),
            duration_ms=round((time.perf_counter() - source_started_at) * 1000, 2),
            status="success",
        )

    ifixit_path = find_first_existing_path(settings.data_dir, IFIXIT_FILE_CANDIDATES)
    if ifixit_path:
        source_started_at = time.perf_counter()
        log_event(
            logger,
            "source_fetch_started",
            source="ifixit",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(ifixit_path),
            status="started",
        )
        ifixit_records = [normalize_ifixit(row) for row in load_jsonl(ifixit_path)]
        normalized_records.extend(ifixit_records)
        source_counts["ifixit"] = len(ifixit_records)
        available_sources.append(("ifixit", ifixit_path))
        log_event(
            logger,
            "source_fetch_completed",
            source="ifixit",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(ifixit_path),
            record_count=len(ifixit_records),
            duration_ms=round((time.perf_counter() - source_started_at) * 1000, 2),
            status="success",
        )

    youtube_path = find_first_existing_path(settings.data_dir, YOUTUBE_FILE_CANDIDATES)
    if youtube_path:
        source_started_at = time.perf_counter()
        log_event(
            logger,
            "source_fetch_started",
            source="youtube",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(youtube_path),
            status="started",
        )
        youtube_records = [normalize_youtube(row) for row in load_jsonl(youtube_path)]
        normalized_records.extend(youtube_records)
        source_counts["youtube"] = len(youtube_records)
        available_sources.append(("youtube", youtube_path))
        log_event(
            logger,
            "source_fetch_completed",
            source="youtube",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(youtube_path),
            record_count=len(youtube_records),
            duration_ms=round((time.perf_counter() - source_started_at) * 1000, 2),
            status="success",
        )

    if include_optional_sources:
        reddit_path = find_first_existing_path(settings.data_dir, REDDIT_FILE_CANDIDATES)
        if reddit_path:
            reddit_records = [normalize_reddit(row) for row in load_jsonl(reddit_path)]
            normalized_records.extend(reddit_records)
            source_counts["reddit"] = len(reddit_records)
            available_sources.append(("reddit", reddit_path))

    if not normalized_records:
        log_event(
            logger,
            "normalization_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="failed",
            error_type="MissingInputError",
            error_message="No real input source files were available for normalization.",
        )
        raise RuntimeError("No real input source files were available for normalization.")

    if publish_raw_snapshots:
        for source_name, source_path in available_sources:
            _upload_raw_snapshot(storage_manager, logger, run_context, source_name, source_path)
            raw_sources_for_promotion.append(source_name)

    _write_jsonl(settings.normalized_jsonl_path, normalized_records)
    _publish_normalized_output(storage_manager, logger, run_context, settings.normalized_jsonl_path)

    rated_values = [
        record["rating_normalized"]
        for record in normalized_records
        if record["rating_normalized"] is not None
    ]

    for source_name in raw_sources_for_promotion:
        current_prefix = storage_manager.resolver.raw_current_prefix(source_name)
        run_prefix = storage_manager.resolver.raw_run_prefix(source_name, run_context.run_id)
        promotion = storage_manager.promote_run_prefix(
            run_prefix,
            current_prefix,
            run_id=run_context.run_id,
            metadata={"source": source_name, "stage": "raw_source_snapshot", "status": "success"},
        )
        log_event(
            logger,
            "latest_run_promoted",
            source=source_name,
            stage="raw_source_snapshot",
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            output_path=current_prefix,
            file_count=promotion["copied_count"],
            status="success",
        )

    log_event(
        logger,
        "normalization_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=len(normalized_records),
        output_path=str(settings.normalized_jsonl_path),
        status="success",
        average_rating_normalized=round(_average(rated_values), 4),
    )

    return normalized_records


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("normalization.local")
    run_context = build_run_context(stage="normalize_local_preview")
    started_at = time.perf_counter()

    log_event(
        logger,
        "pipeline_run_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        status="started",
    )

    try:
        run_local_normalization(run_context=run_context, logger=logger)
        log_event(
            logger,
            "pipeline_run_completed",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
    except Exception as error:
        log_event(
            logger,
            "pipeline_run_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise


if __name__ == "__main__":
    main()
