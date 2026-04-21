"""Cross-source comparison helpers for ReviewPulse AI."""
 
from __future__ import annotations
 
from typing import Any, Mapping, Sequence
 
from src.insights.data_profile import (
    PROPOSAL_SCHEMA_FIELDS,
    coerce_float,
    group_rows_by_source,
    is_missing_value,
    safe_mean,
    word_count_for_row,
)
 
 
def _source_missing_ratio(rows: Sequence[Mapping[str, Any]]) -> float:
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
 
 
def build_source_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
 
    for source_name, source_rows in sorted(group_rows_by_source(rows).items()):
        ratings = [
            value
            for row in source_rows
            if (value := coerce_float(row.get("rating_normalized"))) is not None
        ]
        review_lengths = [float(word_count_for_row(row)) for row in source_rows]
 
        comparisons.append(
            {
                "source": source_name,
                "record_count": len(source_rows),
                "average_review_length_words": round(safe_mean(review_lengths) or 0.0, 2),
                "average_rating_normalized": safe_mean(ratings),
                "missing_value_ratio": _source_missing_ratio(source_rows),
                "missing_rating_ratio": round(
                    sum(1 for row in source_rows if is_missing_value(row.get("rating_normalized"))) / len(source_rows),
                    4,
                ) if source_rows else 0.0,
                "missing_review_text_ratio": round(
                    sum(1 for row in source_rows if is_missing_value(row.get("review_text"))) / len(source_rows),
                    4,
                ) if source_rows else 0.0,
            }
        )
 
    return {"sources": comparisons}
 
 