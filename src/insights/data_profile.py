"""Dataset profiling helpers for ReviewPulse AI."""
 
from __future__ import annotations
 
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence
 
 
PROPOSAL_SCHEMA_FIELDS = (
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
)
 
 
def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False
 
 
def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
 
 
def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)
 
 
def coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
 
    text = str(value).strip()
    if not text:
        return None
 
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
    except ValueError:
        return None
 
 
def format_datetime(value: Any) -> str | None:
    coerced = coerce_datetime(value)
    if coerced is None:
        return None
    return coerced.isoformat()
 
 
def word_count_for_row(row: Mapping[str, Any]) -> int:
    existing = row.get("text_length_words")
    if isinstance(existing, (int, float)) and not isinstance(existing, bool):
        return int(existing)
 
    text = str(row.get("review_text") or "")
    return len([token for token in text.split() if token])
 
 
def group_rows_by_source(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source") or "unknown").strip().lower() or "unknown"
        grouped[source].append(row)
    return dict(grouped)
 
 
def missing_value_ratio(rows: Sequence[Mapping[str, Any]]) -> float:
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
 
 
def _null_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        field_name: sum(1 for row in rows if is_missing_value(row.get(field_name)))
        for field_name in PROPOSAL_SCHEMA_FIELDS
    }
 
 
def _distinct_count(rows: Sequence[Mapping[str, Any]], field_name: str) -> int:
    values = {
        str(row.get(field_name)).strip()
        for row in rows
        if not is_missing_value(row.get(field_name))
    }
    return len(values)
 
 
def _rating_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    rating_values = [
        value
        for row in rows
        if (value := coerce_float(row.get("rating_normalized"))) is not None
    ]
    if not rating_values:
        return {"min": None, "max": None, "mean": None}
 
    return {
        "min": round(min(rating_values), 4),
        "max": round(max(rating_values), 4),
        "mean": safe_mean(rating_values),
    }
 
 
def _date_range(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    dates = [
        date_value
        for row in rows
        if (date_value := coerce_datetime(row.get("review_date"))) is not None
    ]
    if not dates:
        return {"min": None, "max": None}
 
    return {
        "min": min(dates).isoformat(),
        "max": max(dates).isoformat(),
    }
 
 
def _profile_slice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text_lengths = [word_count_for_row(row) for row in rows]
    average_review_length = safe_mean([float(length) for length in text_lengths]) or 0.0
 
    return {
        "record_count": len(rows),
        "null_counts": _null_counts(rows),
        "average_review_length_words": round(average_review_length, 2),
        "rating_normalized": _rating_summary(rows),
        "date_range": _date_range(rows),
        "distinct_products": _distinct_count(rows, "product_name"),
        "distinct_reviewers": _distinct_count(rows, "reviewer_id"),
        "missing_value_ratio": missing_value_ratio(rows),
    }
 
 
def build_dataset_profile(
    rows: Sequence[Mapping[str, Any]],
    dataset_path: str | None = None,
) -> dict[str, Any]:
    grouped = group_rows_by_source(rows)
    profile = _profile_slice(rows)
    profile["dataset_path"] = dataset_path
    profile["source_count"] = len(grouped)
    profile["sources"] = sorted(grouped)
    profile["per_source"] = [
        {
            "source": source_name,
            **_profile_slice(source_rows),
        }
        for source_name, source_rows in sorted(grouped.items())
    ]
    return profile
 