"""PyArrow-backed aggregate metrics for API data-insight endpoints."""
 
from __future__ import annotations
 
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
 
import pyarrow as pa
import pyarrow.dataset as ds
 
from src.insights.data_profile import PROPOSAL_SCHEMA_FIELDS
from src.insights.quality_metrics import normalized_text_key
 
 
WORD_RE = re.compile(r"\S+")
 
 
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False
 
 
def _coerce_datetime(value: Any) -> datetime | None:
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
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
 
 
def _word_count(row: dict[str, Any]) -> int:
    existing = row.get("text_length_words")
    if isinstance(existing, (int, float)) and not isinstance(existing, bool):
        return int(existing)
    return len(WORD_RE.findall(str(row.get("review_text") or "")))
 
 
@dataclass
class _MetricAccumulator:
    record_count: int = 0
    null_counts: dict[str, int] = field(
        default_factory=lambda: {field_name: 0 for field_name in PROPOSAL_SCHEMA_FIELDS}
    )
    total_review_length_words: int = 0
    rating_count: int = 0
    rating_sum: float = 0.0
    rating_min: float | None = None
    rating_max: float | None = None
    date_min: datetime | None = None
    date_max: datetime | None = None
    product_names: set[str] = field(default_factory=set)
    reviewer_ids: set[str] = field(default_factory=set)
    duplicate_seen: set[str] = field(default_factory=set)
    duplicate_count: int = 0
    empty_text_count: int = 0
    invalid_date_count: int = 0
    missing_rating_count: int = 0
    missing_review_text_count: int = 0
 
    def add_row(self, row: dict[str, Any]) -> None:
        self.record_count += 1
 
        for field_name in PROPOSAL_SCHEMA_FIELDS:
            if _is_missing(row.get(field_name)):
                self.null_counts[field_name] += 1
 
        self.total_review_length_words += _word_count(row)
 
        rating = row.get("rating_normalized")
        if _is_missing(rating):
            self.missing_rating_count += 1
        else:
            try:
                rating_value = float(rating)
            except (TypeError, ValueError):
                rating_value = None
            if rating_value is None:
                self.missing_rating_count += 1
            else:
                self.rating_count += 1
                self.rating_sum += rating_value
                self.rating_min = (
                    rating_value
                    if self.rating_min is None
                    else min(self.rating_min, rating_value)
                )
                self.rating_max = (
                    rating_value
                    if self.rating_max is None
                    else max(self.rating_max, rating_value)
                )
 
        review_date = row.get("review_date")
        parsed_date = _coerce_datetime(review_date)
        if not _is_missing(review_date) and parsed_date is None:
            self.invalid_date_count += 1
        if parsed_date is not None:
            self.date_min = parsed_date if self.date_min is None else min(self.date_min, parsed_date)
            self.date_max = parsed_date if self.date_max is None else max(self.date_max, parsed_date)
 
        product_name = row.get("product_name")
        if not _is_missing(product_name):
            self.product_names.add(str(product_name).strip())
        reviewer_id = row.get("reviewer_id")
        if not _is_missing(reviewer_id):
            self.reviewer_ids.add(str(reviewer_id).strip())
 
        text_key = normalized_text_key(row.get("review_text"))
        if not text_key:
            self.empty_text_count += 1
            self.missing_review_text_count += 1
        elif text_key in self.duplicate_seen:
            self.duplicate_count += 1
        else:
            self.duplicate_seen.add(text_key)
 
    def profile_payload(self) -> dict[str, Any]:
        record_count = self.record_count
        total_cells = record_count * len(PROPOSAL_SCHEMA_FIELDS)
        rating_summary = {
            "min": round(self.rating_min, 4) if self.rating_min is not None else None,
            "max": round(self.rating_max, 4) if self.rating_max is not None else None,
            "mean": round(self.rating_sum / self.rating_count, 4)
            if self.rating_count
            else None,
        }
 
        return {
            "record_count": record_count,
            "null_counts": dict(self.null_counts),
            "average_review_length_words": round(
                self.total_review_length_words / record_count,
                2,
            )
            if record_count
            else 0.0,
            "rating_normalized": rating_summary,
            "date_range": {
                "min": self.date_min.isoformat() if self.date_min else None,
                "max": self.date_max.isoformat() if self.date_max else None,
            },
            "distinct_products": len(self.product_names),
            "distinct_reviewers": len(self.reviewer_ids),
            "missing_value_ratio": round(
                sum(self.null_counts.values()) / total_cells,
                4,
            )
            if total_cells
            else 0.0,
        }
 
    def source_comparison_payload(self, source_name: str) -> dict[str, Any]:
        record_count = self.record_count
        total_cells = record_count * len(PROPOSAL_SCHEMA_FIELDS)
        return {
            "source": source_name,
            "record_count": record_count,
            "average_review_length_words": round(
                self.total_review_length_words / record_count,
                2,
            )
            if record_count
            else 0.0,
            "average_rating_normalized": round(self.rating_sum / self.rating_count, 4)
            if self.rating_count
            else None,
            "missing_value_ratio": round(
                sum(self.null_counts.values()) / total_cells,
                4,
            )
            if total_cells
            else 0.0,
            "missing_rating_ratio": round(self.missing_rating_count / record_count, 4)
            if record_count
            else 0.0,
            "missing_review_text_ratio": round(
                self.missing_review_text_count / record_count,
                4,
            )
            if record_count
            else 0.0,
        }
 
    def quality_payload(self) -> dict[str, Any]:
        record_count = self.record_count
        total_cells = record_count * len(PROPOSAL_SCHEMA_FIELDS)
        return {
            "record_count": record_count,
            "duplicate_count": self.duplicate_count,
            "duplicate_ratio": round(self.duplicate_count / record_count, 4)
            if record_count
            else 0.0,
            "null_ratio": round(sum(self.null_counts.values()) / total_cells, 4)
            if total_cells
            else 0.0,
            "empty_text_count": self.empty_text_count,
            "empty_text_ratio": round(self.empty_text_count / record_count, 4)
            if record_count
            else 0.0,
            "invalid_date_count": self.invalid_date_count,
            "invalid_date_ratio": round(self.invalid_date_count / record_count, 4)
            if record_count
            else 0.0,
            "missing_rating_count": self.missing_rating_count,
            "missing_rating_ratio": round(self.missing_rating_count / record_count, 4)
            if record_count
            else 0.0,
        }
 
 
def _read_metric_accumulators(
    dataset_path: Path,
) -> tuple[_MetricAccumulator, dict[str, _MetricAccumulator]]:
    dataset = ds.dataset(str(dataset_path), format="parquet")
    available_columns = set(dataset.schema.names)
    requested_columns = sorted(
        (
            set(PROPOSAL_SCHEMA_FIELDS)
            | {"text_length_words"}
        )
        & available_columns
    )
 
    overall = _MetricAccumulator()
    by_source: dict[str, _MetricAccumulator] = {}
 
    for record_batch in dataset.to_batches(
        columns=requested_columns,
        batch_size=100000,
    ):
        rows = pa.Table.from_batches([record_batch]).to_pylist()
        for row in rows:
            source_name = str(row.get("source") or "unknown").strip().lower() or "unknown"
            source_accumulator = by_source.setdefault(source_name, _MetricAccumulator())
            overall.add_row(row)
            source_accumulator.add_row(row)
 
    return overall, by_source
 
 
def build_arrow_dataset_profile(dataset_path: Path) -> dict[str, Any]:
    overall, by_source = _read_metric_accumulators(dataset_path)
    profile = overall.profile_payload()
    profile["dataset_path"] = str(dataset_path)
    profile["source_count"] = len(by_source)
    profile["sources"] = sorted(by_source)
    profile["per_source"] = [
        {"source": source_name, **accumulator.profile_payload()}
        for source_name, accumulator in sorted(by_source.items())
    ]
    return profile
 
 
def build_arrow_source_comparison(dataset_path: Path) -> dict[str, Any]:
    _overall, by_source = _read_metric_accumulators(dataset_path)
    return {
        "sources": [
            accumulator.source_comparison_payload(source_name)
            for source_name, accumulator in sorted(by_source.items())
        ]
    }
 
 
def build_arrow_quality_metrics(dataset_path: Path) -> dict[str, Any]:
    overall, by_source = _read_metric_accumulators(dataset_path)
    metrics = overall.quality_payload()
    metrics["duplicate_method"] = (
        "Exact duplicate detection on lower-cased, whitespace-normalized review_text. "
        "Near-duplicate MinHash/LSH remains a later milestone."
    )
    metrics["per_source"] = [
        {"source": source_name, **accumulator.quality_payload()}
        for source_name, accumulator in sorted(by_source.items())
    ]
    return metrics
 
 
def build_arrow_data_insights(dataset_path: Path) -> dict[str, dict[str, Any]]:
    overall, by_source = _read_metric_accumulators(dataset_path)
 
    profile = overall.profile_payload()
    profile["dataset_path"] = str(dataset_path)
    profile["source_count"] = len(by_source)
    profile["sources"] = sorted(by_source)
    profile["per_source"] = [
        {"source": source_name, **accumulator.profile_payload()}
        for source_name, accumulator in sorted(by_source.items())
    ]
 
    comparison = {
        "sources": [
            accumulator.source_comparison_payload(source_name)
            for source_name, accumulator in sorted(by_source.items())
        ],
        "dataset_path": str(dataset_path),
    }
 
    quality = overall.quality_payload()
    quality["duplicate_method"] = (
        "Exact duplicate detection on lower-cased, whitespace-normalized review_text. "
        "Near-duplicate MinHash/LSH remains a later milestone."
    )
    quality["per_source"] = [
        {"source": source_name, **accumulator.quality_payload()}
        for source_name, accumulator in sorted(by_source.items())
    ]
    quality["dataset_path"] = str(dataset_path)
 
    return {
        "profile": profile,
        "comparison": comparison,
        "quality": quality,
    }
 
 