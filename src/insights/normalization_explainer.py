"""Normalization explanations for supported data sources."""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Any
 
from src.common.settings import get_settings
from src.normalization.core import (
    AMAZON_FILE_CANDIDATES,
    EBAY_FILE_CANDIDATES,
    IFIXIT_FILE_CANDIDATES,
    YELP_REVIEW_FILE_CANDIDATES,
    YOUTUBE_FILE_CANDIDATES,
    find_first_existing_path,
    load_jsonl,
    normalize_amazon,
    normalize_ebay,
    normalize_ifixit,
    normalize_yelp,
    normalize_youtube,
    resolve_source_input_path,
    resolve_yelp_source_paths,
)
 
 
SUPPORTED_SOURCES = ("amazon", "yelp", "ebay", "ifixit", "youtube")
 
SOURCE_RULES: dict[str, dict[str, str]] = {
    "amazon": {
        "raw_rating_field": "rating",
        "raw_date_field": "timestamp",
        "raw_scale": "1.0 to 5.0 stars",
        "formula": "(rating - 1.0) / 4.0",
        "rating_notes": "Amazon reviews arrive on a 1 to 5 star scale and are compressed into the shared 0 to 1 scale.",
        "date_notes": "Amazon stores review time in Unix milliseconds, which is converted to ISO 8601 UTC.",
    },
    "yelp": {
        "raw_rating_field": "stars",
        "raw_date_field": "date",
        "raw_scale": "1 to 5 stars",
        "formula": "(stars - 1.0) / 4.0",
        "rating_notes": "Yelp stars already match a 1 to 5 scale, so the same 0 to 1 transformation applies.",
        "date_notes": "Yelp review dates are stored as calendar dates and normalized to ISO 8601 at midnight.",
    },
    "ebay": {
        "raw_rating_field": "seller_rating",
        "raw_date_field": "listing_date",
        "raw_scale": "0 to 100 seller percentage",
        "formula": "seller_rating / 100.0",
        "rating_notes": "eBay uses seller percentages rather than stars, so the normalized score is the percentage divided by 100.",
        "date_notes": "eBay listing or review timestamps are already ISO-like values and are preserved in ISO 8601 form.",
    },
    "ifixit": {
        "raw_rating_field": "repairability_score",
        "raw_date_field": "published_date",
        "raw_scale": "1 to 10 repairability score",
        "formula": "(repairability_score - 1.0) / 9.0",
        "rating_notes": "iFixit repairability scores span 1 to 10, so the shared scale subtracts 1 and divides by 9.",
        "date_notes": "iFixit publication dates are preserved as ISO 8601 timestamps when available.",
    },
    "youtube": {
        "raw_rating_field": "rating",
        "raw_date_field": "published_date or created_utc",
        "raw_scale": "No native rating field",
        "formula": "rating_normalized = null",
        "rating_notes": "YouTube transcripts do not carry a native rating, so rating_normalized remains null and sentiment is derived later from text.",
        "date_notes": "YouTube uses transcript metadata or publication timestamps, which are normalized into ISO 8601 when present.",
    },
}
 
 
def _source_candidates(source: str) -> tuple[str, ...]:
    mapping = {
        "amazon": AMAZON_FILE_CANDIDATES,
        "yelp": YELP_REVIEW_FILE_CANDIDATES,
        "ebay": EBAY_FILE_CANDIDATES,
        "ifixit": IFIXIT_FILE_CANDIDATES,
        "youtube": YOUTUBE_FILE_CANDIDATES,
    }
    return mapping[source]
 
 
def _normalize_source_row(source: str, raw_row: dict[str, Any]) -> dict[str, Any]:
    if source == "amazon":
        return normalize_amazon(raw_row)
    if source == "yelp":
        return normalize_yelp(raw_row)
    if source == "ebay":
        return normalize_ebay(raw_row)
    if source == "ifixit":
        return normalize_ifixit(raw_row)
    if source == "youtube":
        return normalize_youtube(raw_row)
    raise ValueError(f"Unsupported source: {source}")
 
 
def _load_sample_raw_row(source: str, sample_index: int = 0) -> tuple[dict[str, Any] | None, Path | None]:
    settings = get_settings()
    if source == "yelp":
        raw_path, _ = resolve_yelp_source_paths(settings.data_dir)
    else:
        raw_path = resolve_source_input_path(settings.data_dir, source, _source_candidates(source))
    if raw_path is None:
        return None, None
 
    rows = load_jsonl(raw_path, limit=sample_index + 1)
    if sample_index >= len(rows):
        return None, raw_path
    return rows[sample_index], raw_path
 
 
def _build_explanation(source: str, sample_index: int) -> dict[str, Any]:
    rule = SOURCE_RULES[source]
    raw_row, raw_path = _load_sample_raw_row(source, sample_index=sample_index)
 
    explanation: dict[str, Any] = {
        "source": source,
        "raw_rating_field": rule["raw_rating_field"],
        "raw_date_field": rule["raw_date_field"],
        "raw_scale": rule["raw_scale"],
        "normalized_scale": "0 to 1 shared rating range",
        "formula": rule["formula"],
        "rating_notes": rule["rating_notes"],
        "date_notes": rule["date_notes"],
        "sample_found": raw_row is not None,
        "sample_source_path": str(raw_path) if raw_path else None,
        "sample_index": sample_index,
        "sample_identifiers": None,
        "sample_raw_rating_value": None,
        "sample_normalized_rating": None,
        "sample_raw_date_value": None,
        "sample_normalized_review_date": None,
    }
 
    if raw_row is None:
        return explanation
 
    normalized_row = _normalize_source_row(source, raw_row)
    explanation["sample_identifiers"] = {
        "review_id": normalized_row.get("review_id"),
        "product_name": normalized_row.get("product_name"),
        "reviewer_id": normalized_row.get("reviewer_id"),
    }
    explanation["sample_raw_rating_value"] = raw_row.get(rule["raw_rating_field"])
    explanation["sample_normalized_rating"] = normalized_row.get("rating_normalized")
    explanation["sample_normalized_review_date"] = normalized_row.get("review_date")
 
    if source == "youtube":
        explanation["sample_raw_date_value"] = raw_row.get("published_date") or raw_row.get("created_utc")
    else:
        explanation["sample_raw_date_value"] = raw_row.get(rule["raw_date_field"])
 
    return explanation
 
 
def build_normalization_explanations(
    source: str | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    if source is not None:
        normalized_source = source.strip().lower()
        if normalized_source not in SUPPORTED_SOURCES:
            return {
                "error": (
                    "Unsupported source. Choose one of: "
                    + ", ".join(SUPPORTED_SOURCES)
                )
            }
        sources = [normalized_source]
    else:
        sources = list(SUPPORTED_SOURCES)
 
    return {
        "explanations": [
            _build_explanation(source_name, sample_index=sample_index)
            for source_name in sources
        ]
    }
 
 
