"""Data quality metrics for normalized ReviewPulse datasets."""
 
from __future__ import annotations
 
from typing import Any, Mapping, Sequence
 
from src.insights.data_profile import (
    PROPOSAL_SCHEMA_FIELDS,
    coerce_datetime,
    group_rows_by_source,
    is_missing_value,
)
 
 
def normalized_text_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.split())
 
 
def _duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
 
    for row in rows:
        key = normalized_text_key(row.get("review_text"))
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
 
    return duplicates
 
 
def _null_ratio(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
 
    missing_cells = sum(
        1
        for row in rows
        for field_name in PROPOSAL_SCHEMA_FIELDS
        if is_missing_value(row.get(field_name))
    )
    total_cells = len(rows) * len(PROPOSAL_SCHEMA_FIELDS)
    return round(missing_cells / total_cells, 4) if total_cells else 0.0
 
 
def _quality_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record_count = len(rows)
    duplicate_count = _duplicate_count(rows)
    empty_text_count = sum(1 for row in rows if not normalized_text_key(row.get("review_text")))
    invalid_date_count = sum(
        1
        for row in rows
        if not is_missing_value(row.get("review_date")) and coerce_datetime(row.get("review_date")) is None
    )
    missing_rating_count = sum(1 for row in rows if is_missing_value(row.get("rating_normalized")))
 
    return {
        "record_count": record_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": round(duplicate_count / record_count, 4) if record_count else 0.0,
        "null_ratio": _null_ratio(rows),
        "empty_text_count": empty_text_count,
        "empty_text_ratio": round(empty_text_count / record_count, 4) if record_count else 0.0,
        "invalid_date_count": invalid_date_count,
        "invalid_date_ratio": round(invalid_date_count / record_count, 4) if record_count else 0.0,
        "missing_rating_count": missing_rating_count,
        "missing_rating_ratio": round(missing_rating_count / record_count, 4) if record_count else 0.0,
    }
 
 
def build_quality_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = group_rows_by_source(rows)
    metrics = _quality_slice(rows)
    metrics["duplicate_method"] = (
        "Exact duplicate detection on lower-cased, whitespace-normalized review_text. "
        "Near-duplicate MinHash/LSH remains a later milestone."
    )
    metrics["per_source"] = [
        {
            "source": source_name,
            **_quality_slice(source_rows),
        }
        for source_name, source_rows in sorted(grouped.items())
    ]
    return metrics
 
 