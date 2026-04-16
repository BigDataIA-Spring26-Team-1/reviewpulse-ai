"""Unified schema and source normalization logic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.common.settings import get_settings


PROPOSAL_CORE_SOURCES = ("amazon", "yelp", "ebay", "ifixit", "youtube")
OPTIONAL_SOURCES = ("reddit",)

AMAZON_FILE_CANDIDATES = (
    "amazon_electronics_sample.jsonl",
    "amazon/amazon_electronics_sample.jsonl",
)
YELP_REVIEW_FILE_CANDIDATES = (
    "yelp/yelp_academic_dataset_review.json",
    "yelp_reviews.jsonl",
)
YELP_BUSINESS_FILE_CANDIDATES = (
    "yelp/yelp_academic_dataset_business.json",
    "yelp_business.jsonl",
)
EBAY_FILE_CANDIDATES = (
    "ebay_reviews.jsonl",
    "ebay/ebay_reviews.jsonl",
    "ebay/ebay_listings.jsonl",
    "ebay/listings.jsonl",
)
IFIXIT_FILE_CANDIDATES = (
    "ifixit_reviews.jsonl",
    "ifixit/ifixit_reviews.jsonl",
    "ifixit/guides.jsonl",
)
YOUTUBE_FILE_CANDIDATES = (
    "youtube_reviews.jsonl",
    "youtube/youtube_reviews.jsonl",
)
REDDIT_FILE_CANDIDATES = (
    "reddit_reviews.jsonl",
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
        review_date=isoformat_from_datetime_string(raw.get("published_date")),
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
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(json.loads(payload))
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


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_local_normalization(include_optional_sources: bool = True) -> list[dict[str, Any]]:
    settings = get_settings()
    normalized_records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    amazon_path = find_first_existing_path(settings.data_dir, AMAZON_FILE_CANDIDATES)
    if amazon_path:
        amazon_records = [normalize_amazon(row) for row in load_jsonl(amazon_path)]
        normalized_records.extend(amazon_records)
        source_counts["amazon"] = len(amazon_records)
        print(f"Loaded {len(amazon_records)} Amazon records from {amazon_path}")
    else:
        print("Amazon input not found.")

    yelp_review_path = find_first_existing_path(settings.data_dir, YELP_REVIEW_FILE_CANDIDATES)
    if yelp_review_path:
        yelp_business_path = find_first_existing_path(settings.data_dir, YELP_BUSINESS_FILE_CANDIDATES)
        business_lookup = load_yelp_business_lookup(yelp_business_path) if yelp_business_path else {}
        yelp_records = [
            normalize_yelp(row, business_lookup)
            for row in load_jsonl(yelp_review_path)
        ]
        normalized_records.extend(yelp_records)
        source_counts["yelp"] = len(yelp_records)
        print(f"Loaded {len(yelp_records)} Yelp records from {yelp_review_path}")
    else:
        print("Yelp input not found.")

    ebay_path = find_first_existing_path(settings.data_dir, EBAY_FILE_CANDIDATES)
    if ebay_path:
        ebay_records = [normalize_ebay(row) for row in load_jsonl(ebay_path)]
        normalized_records.extend(ebay_records)
        source_counts["ebay"] = len(ebay_records)
        print(f"Loaded {len(ebay_records)} eBay records from {ebay_path}")
    else:
        print("eBay input not found.")

    ifixit_path = find_first_existing_path(settings.data_dir, IFIXIT_FILE_CANDIDATES)
    if ifixit_path:
        ifixit_records = [normalize_ifixit(row) for row in load_jsonl(ifixit_path)]
        normalized_records.extend(ifixit_records)
        source_counts["ifixit"] = len(ifixit_records)
        print(f"Loaded {len(ifixit_records)} iFixit records from {ifixit_path}")
    else:
        print("iFixit input not found.")

    youtube_path = find_first_existing_path(settings.data_dir, YOUTUBE_FILE_CANDIDATES)
    if youtube_path:
        youtube_records = [normalize_youtube(row) for row in load_jsonl(youtube_path)]
        normalized_records.extend(youtube_records)
        source_counts["youtube"] = len(youtube_records)
        print(f"Loaded {len(youtube_records)} YouTube records from {youtube_path}")
    else:
        print("YouTube input not found.")

    if include_optional_sources:
        reddit_path = find_first_existing_path(settings.data_dir, REDDIT_FILE_CANDIDATES)
        if reddit_path:
            reddit_records = [normalize_reddit(row) for row in load_jsonl(reddit_path)]
            normalized_records.extend(reddit_records)
            source_counts["reddit"] = len(reddit_records)
            print(f"Loaded {len(reddit_records)} Reddit records from {reddit_path}")

    if not normalized_records:
        print("No source files were available for normalization.")
        return []

    _write_jsonl(settings.normalized_jsonl_path, normalized_records)

    print("\nNormalization report")
    print(f"Total normalized records: {len(normalized_records)}")
    for source_name in PROPOSAL_CORE_SOURCES + OPTIONAL_SOURCES:
        if source_name in source_counts:
            print(f"  {source_name}: {source_counts[source_name]}")

    rated_values = [
        record["rating_normalized"]
        for record in normalized_records
        if record["rating_normalized"] is not None
    ]
    print(f"Average normalized rating: {_average(rated_values):.4f}")
    print(f"Output written to: {settings.normalized_jsonl_path}")

    return normalized_records


def main() -> None:
    run_local_normalization()


if __name__ == "__main__":
    main()
