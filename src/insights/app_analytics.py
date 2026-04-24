"""Application-facing analytics helpers over ReviewPulse parquet artifacts."""
 
from __future__ import annotations
 
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
 
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
 
from src.common.amazon_titles import resolve_amazon_product_titles
from src.retrieval.sqlite_vector_store import sqlite_store_path
 
 
APP_REVIEW_COLUMNS = (
    "review_id",
    "product_name",
    "product_title",
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
    "aspect_labels",
    "aspect_count",
    "aspect_details_json",
    "sentiment_label",
    "sentiment_score",
)
SOURCE_NAMES = ("amazon", "ebay", "yelp", "ifixit", "youtube", "reddit")
RATING_BINS = (
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, 0.6, "0.4-0.6"),
    (0.6, 0.8, "0.6-0.8"),
    (0.8, 1.01, "0.8-1.0"),
)
WORD_RE = re.compile(r"\S+")
MACHINE_ID_RE = re.compile(r"[A-Za-z0-9_-]{10,}")
 
 
@dataclass(frozen=True, slots=True)
class ReviewFilter:
    query: str | None = None
    source: str | None = None
    sentiment: str | None = None
    product: str | None = None
    category: str | None = None
    date_start: datetime | None = None
    date_end: datetime | None = None
    min_rating: float | None = None
    max_rating: float | None = None
 
 
def _requested_columns(dataset: ds.Dataset, extra_columns: Iterable[str] = ()) -> list[str]:
    available = set(dataset.schema.names)
    columns = [column for column in APP_REVIEW_COLUMNS if column in available]
    for column in extra_columns:
        if column in available and column not in columns:
            columns.append(column)
    return columns
 
 
def _iter_rows(
    dataset_path: Path,
    *,
    columns: Iterable[str] = APP_REVIEW_COLUMNS,
    batch_size: int = 100000,
    max_rows: int | None = None,
) -> Iterable[dict[str, Any]]:
    dataset = ds.dataset(str(dataset_path), format="parquet")
    available = set(dataset.schema.names)
    requested = [column for column in columns if column in available]
    if not requested:
        return
    effective_batch_size = min(batch_size, max_rows) if max_rows else batch_size
    yielded_rows = 0
    for record_batch in dataset.to_batches(columns=requested, batch_size=effective_batch_size):
        for row in pa.Table.from_batches([record_batch]).to_pylist():
            if max_rows is not None and yielded_rows >= max_rows:
                return
            yielded_rows += 1
            yield dict(row)
 
 
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
 
 
def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float):
        return None
    return value_float
 
 
def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()
 
 
def _looks_like_machine_id(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return bool(MACHINE_ID_RE.fullmatch(text))
 
 
def _split_aspects(value: Any) -> list[str]:
    return [
        aspect.strip().lower()
        for aspect in re.split(r"\s*,\s*", str(value or ""))
        if aspect and aspect.strip()
    ]
 
 
def _month_key(value: Any) -> str | None:
    parsed = _coerce_datetime(value)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m")
 
 
def _rating_bin(value: Any) -> str | None:
    rating = _coerce_float(value)
    if rating is None:
        return None
    for lower, upper, label in RATING_BINS:
        if lower <= rating < upper:
            return label
    return None
 
 
def _clean_product_label_candidate(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if _looks_like_machine_id(text):
        return None
 
    lowered = text.lower()
    if lowered.startswith("amazon electronics item "):
        return None
    if lowered in {
        "amazon product review",
        "yelp business review",
        "ebay listing",
        "ifixit guide",
        "reddit discussion",
        "youtube review",
        "unknown item",
        "unknown",
    }:
        return None
    return text
 
 
def _product_title_from_source_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
 
    parsed = urlparse(text)
    path_parts = [part.strip() for part in parsed.path.split("/") if part.strip()]
    if not path_parts:
        return None
 
    if "dp" in path_parts:
        dp_index = path_parts.index("dp")
        if dp_index > 0:
            slug = re.sub(r"[-_]+", " ", path_parts[dp_index - 1]).strip()
            slug = re.sub(r"\s+", " ", slug)
            normalized_slug = slug.replace(" ", "")
            if slug and not _looks_like_machine_id(normalized_slug):
                return slug
    return None
 
 
def _fallback_product_label(source: str, product_name: str) -> str:
    if not product_name:
        return "Unknown Product"
    if source == "amazon":
        return f"Amazon Product (ASIN {product_name})"
    if source == "yelp":
        return f"Yelp Business ({product_name})"
    if source == "ebay":
        return f"eBay Listing ({product_name})"
    if source == "ifixit":
        return f"iFixit Guide ({product_name})"
    if source == "youtube":
        return f"YouTube Review ({product_name})"
    if source == "reddit":
        return f"Reddit Discussion ({product_name})"
    return product_name
 
 
def _product_label_from_components(
    *,
    source: Any,
    product_name: Any,
    product_title: Any = None,
    display_name_counts: Counter[str] | None = None,
    source_url_counts: Counter[str] | None = None,
) -> str:
    source_text = str(source or "").strip().lower()
    product_name_text = str(product_name or "").strip()
    product_title_text = _clean_product_label_candidate(product_title)
 
    if product_title_text:
        return product_title_text
 
    if product_name_text and not _looks_like_machine_id(product_name_text):
        return product_name_text
 
    if source_url_counts:
        for source_url, _count in source_url_counts.most_common():
            title_from_url = _product_title_from_source_url(source_url)
            if title_from_url:
                return title_from_url
 
    if display_name_counts:
        candidate, count = display_name_counts.most_common(1)[0]
        total = sum(display_name_counts.values())
        if count >= 2 and total > 0 and (count / total) >= 0.6:
            return candidate
 
    return _fallback_product_label(source_text, product_name_text)
 
 
def _product_label_for_row(row: dict[str, Any], *, amazon_titles: dict[str, str] | None = None) -> str:
    source_url = str(row.get("source_url") or "").strip()
    source_url_counts = Counter({source_url: 1}) if source_url else Counter()
    product_name = str(row.get("product_name") or "").strip()
    product_title = row.get("product_title")
    if not product_title and str(row.get("source") or "").strip().lower() == "amazon" and amazon_titles:
        product_title = amazon_titles.get(product_name)
    return _product_label_from_components(
        source=row.get("source"),
        product_name=product_name,
        product_title=product_title,
        source_url_counts=source_url_counts,
    )
 
 
def _matches_filter(row: dict[str, Any], filters: ReviewFilter) -> bool:
    if filters.source and _normalized_text(row.get("source")) != filters.source.lower():
        return False
    if filters.sentiment and _normalized_text(row.get("sentiment_label")) != filters.sentiment.lower():
        return False
    if filters.product and _normalized_text(row.get("product_name")) != filters.product.lower():
        return False
    if filters.category and filters.category.lower() not in _normalized_text(row.get("product_category")):
        return False
 
    rating = _coerce_float(row.get("rating_normalized"))
    if filters.min_rating is not None and (rating is None or rating < filters.min_rating):
        return False
    if filters.max_rating is not None and (rating is None or rating > filters.max_rating):
        return False
 
    review_date = _coerce_datetime(row.get("review_date"))
    if filters.date_start is not None and (review_date is None or review_date < filters.date_start):
        return False
    if filters.date_end is not None and (review_date is None or review_date > filters.date_end):
        return False
 
    if filters.query:
        needle = filters.query.lower()
        searchable_text = " ".join(
            _normalized_text(row.get(column))
            for column in (
                "review_text",
                "product_name",
                "product_title",
                "display_name",
                "product_category",
                "display_category",
                "aspect_labels",
            )
        )
        if needle not in searchable_text:
            return False
    return True
 
 
def _review_payload(row: dict[str, Any], *, amazon_titles: dict[str, str] | None = None) -> dict[str, Any]:
    review_date = _coerce_datetime(row.get("review_date"))
    review_text = str(row.get("review_text") or "")
    product_name = str(row.get("product_name") or "")
    product_title = str(row.get("product_title") or "")
    if not product_title and str(row.get("source") or "").strip().lower() == "amazon" and amazon_titles:
        product_title = str(amazon_titles.get(product_name) or "")
    return {
        "review_id": str(row.get("review_id") or ""),
        "source": str(row.get("source") or ""),
        "product_name": product_name,
        "product_title": product_title,
        "product_label": _product_label_for_row(row, amazon_titles=amazon_titles),
        "product_category": str(row.get("product_category") or ""),
        "display_name": str(row.get("display_name") or product_name or ""),
        "display_category": str(row.get("display_category") or row.get("product_category") or ""),
        "entity_type": str(row.get("entity_type") or "review"),
        "rating_normalized": _coerce_float(row.get("rating_normalized")),
        "sentiment_label": str(row.get("sentiment_label") or ""),
        "sentiment_score": _coerce_float(row.get("sentiment_score")),
        "aspect_labels": str(row.get("aspect_labels") or ""),
        "aspect_count": int(row.get("aspect_count") or 0),
        "review_date": review_date.isoformat() if review_date else None,
        "helpful_votes": int(row.get("helpful_votes") or 0),
        "verified_purchase": row.get("verified_purchase"),
        "source_url": str(row.get("source_url") or ""),
        "review_text": review_text,
        "text_length_words": len(WORD_RE.findall(review_text)),
    }
 
 
def parquet_record_count(dataset_path: Path) -> int:
    if not dataset_path.exists():
        return 0
    if dataset_path.is_file():
        return pq.ParquetFile(dataset_path).metadata.num_rows
    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in dataset_path.rglob("*.parquet")
        if path.is_file()
    )
 
 
def sqlite_vector_count(chroma_path: Path) -> int | None:
    db_path = sqlite_store_path(chroma_path)
    if not db_path.exists():
        return None
    connection = sqlite3.connect(str(db_path))
    try:
        row = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()
    finally:
        connection.close()
    return int(row[0] if row else 0)
 
 
def list_filter_options(dataset_path: Path, *, limit: int = 200, max_rows: int | None = None) -> dict[str, Any]:
    product_counts: Counter[str] = Counter()
    product_source_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    product_display_name_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    product_source_url_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    date_min: datetime | None = None
    date_max: datetime | None = None
    scanned_rows = 0
 
    for row in _iter_rows(
        dataset_path,
        columns=("product_name", "product_category", "source", "sentiment_label", "review_date", "display_name", "source_url"),
        max_rows=max_rows,
    ):
        scanned_rows += 1
        product = str(row.get("product_name") or "").strip()
        category = str(row.get("product_category") or "").strip()
        source = str(row.get("source") or "").strip().lower()
        sentiment = str(row.get("sentiment_label") or "").strip().lower()
        if product:
            product_counts[product] += 1
            if source:
                product_source_counts[product][source] += 1
            display_name = _clean_product_label_candidate(row.get("display_name"))
            if display_name:
                product_display_name_counts[product][display_name] += 1
            source_url = str(row.get("source_url") or "").strip()
            if source_url:
                product_source_url_counts[product][source_url] += 1
        if category:
            category_counts[category] += 1
        if source:
            source_counts[source] += 1
        if sentiment:
            sentiment_counts[sentiment] += 1
        parsed_date = _coerce_datetime(row.get("review_date"))
        if parsed_date is not None:
            date_min = parsed_date if date_min is None else min(date_min, parsed_date)
            date_max = parsed_date if date_max is None else max(date_max, parsed_date)
 
    top_products = product_counts.most_common(limit)
    amazon_titles = resolve_amazon_product_titles(
        [
            name
            for name, _count in top_products
            if product_source_counts[name].most_common(1)
            and product_source_counts[name].most_common(1)[0][0] == "amazon"
            and _looks_like_machine_id(name)
        ]
    )
 
    return {
        "products": [
            {
                "product_name": name,
                "product_title": amazon_titles.get(name, ""),
                "product_label": _product_label_from_components(
                    source=product_source_counts[name].most_common(1)[0][0] if product_source_counts[name] else "",
                    product_name=name,
                    product_title=amazon_titles.get(name),
                    display_name_counts=product_display_name_counts[name],
                    source_url_counts=product_source_url_counts[name],
                ),
                "review_count": count,
            }
            for name, count in top_products
        ],
        "categories": [
            {"product_category": name, "review_count": count}
            for name, count in category_counts.most_common(limit)
        ],
        "sources": [
            {"source": name, "review_count": count}
            for name, count in sorted(source_counts.items())
        ],
        "sentiments": [
            {"sentiment_label": name, "review_count": count}
            for name, count in sorted(sentiment_counts.items())
        ],
        "date_range": {
            "min": date_min.isoformat() if date_min else None,
            "max": date_max.isoformat() if date_max else None,
        },
        "rows_scanned": scanned_rows,
        "scan_limited": max_rows is not None,
        "max_rows": max_rows,
    }
 
 
def explore_reviews(
    dataset_path: Path,
    filters: ReviewFilter,
    *,
    limit: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    rows_raw: list[dict[str, Any]] = []
    matched_count = 0
    requested = max(1, limit)
    skipped = max(0, offset)
 
    for row in _iter_rows(dataset_path):
        if not _matches_filter(row, filters):
            continue
        if matched_count >= skipped and len(rows_raw) < requested:
            rows_raw.append(dict(row))
        matched_count += 1
        if len(rows_raw) >= requested and matched_count > skipped + requested:
            break
 
    amazon_titles = resolve_amazon_product_titles(
        str(row.get("product_name") or "")
        for row in rows_raw
        if str(row.get("source") or "").strip().lower() == "amazon"
        and _looks_like_machine_id(row.get("product_name"))
    )
    rows = [_review_payload(row, amazon_titles=amazon_titles) for row in rows_raw]
 
    return {
        "rows": rows,
        "matched_count": matched_count,
        "limit": requested,
        "offset": skipped,
        "has_more": matched_count > skipped + len(rows),
        "dataset_path": str(dataset_path),
    }
 
 
def build_dashboard_summary(
    *,
    normalized_path: Path,
    sentiment_path: Path,
    chroma_path: Path,
    hitl_queue_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    dataset_path = sentiment_path if sentiment_path.exists() else normalized_path
    record_count = parquet_record_count(dataset_path) if dataset_path.exists() else 0
    vector_count = sqlite_vector_count(chroma_path)
    hitl_count = 0
    if hitl_queue_path.exists():
        hitl_count = sum(1 for line in hitl_queue_path.read_text(encoding="utf-8").splitlines() if line.strip())
 
    source_count = sum(1 for source in SOURCE_NAMES if (data_dir / source).exists())
    latest_dataset_update = (
        datetime.fromtimestamp(dataset_path.stat().st_mtime, UTC).isoformat()
        if dataset_path.exists()
        else None
    )
 
    artifact_dirs = [
        {
            "name": path.name,
            "path": str(path),
            "exists": path.exists(),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            if path.exists()
            else None,
        }
        for path in (
            data_dir / "normalized_reviews_parquet",
            data_dir / "reviews_with_sentiment_parquet",
            chroma_path,
        )
    ]
 
    return {
        "dataset_path": str(dataset_path) if dataset_path.exists() else None,
        "total_reviews": record_count,
        "sources_integrated": source_count,
        "products_tracked": None,
        "positive_sentiment_ratio": None,
        "indexed_documents": vector_count,
        "hitl_queue_items": hitl_count,
        "latest_review_date": None,
        "latest_dataset_update": latest_dataset_update,
        "artifact_dirs": artifact_dirs,
    }
 
 
def build_sentiment_analytics(
    dataset_path: Path,
    filters: ReviewFilter,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    sentiment_counts: Counter[str] = Counter()
    rating_counts: Counter[str] = Counter()
    source_sentiment: Counter[tuple[str, str]] = Counter()
    monthly_sentiment: Counter[tuple[str, str]] = Counter()
    aspect_counts: Counter[str] = Counter()
    reviewed_rows = 0
    scanned_rows = 0
 
    for row in _iter_rows(dataset_path, max_rows=max_rows):
        scanned_rows += 1
        if not _matches_filter(row, filters):
            continue
        reviewed_rows += 1
        source = str(row.get("source") or "unknown").strip().lower() or "unknown"
        sentiment = str(row.get("sentiment_label") or "unknown").strip().lower() or "unknown"
        sentiment_counts[sentiment] += 1
        source_sentiment[(source, sentiment)] += 1
        if rating_bin := _rating_bin(row.get("rating_normalized")):
            rating_counts[rating_bin] += 1
        if month := _month_key(row.get("review_date")):
            monthly_sentiment[(month, sentiment)] += 1
        for aspect in _split_aspects(row.get("aspect_labels")):
            aspect_counts[aspect] += 1
 
    return {
        "record_count": reviewed_rows,
        "sentiment_distribution": [
            {"sentiment_label": sentiment, "count": count}
            for sentiment, count in sorted(sentiment_counts.items())
        ],
        "rating_distribution": [
            {"rating_bucket": label, "count": rating_counts[label]}
            for _lower, _upper, label in RATING_BINS
        ],
        "source_sentiment": [
            {"source": source, "sentiment_label": sentiment, "count": count}
            for (source, sentiment), count in sorted(source_sentiment.items())
        ],
        "monthly_trend": [
            {"month": month, "sentiment_label": sentiment, "count": count}
            for (month, sentiment), count in sorted(monthly_sentiment.items())
        ],
        "top_aspects": [
            {"aspect": aspect, "count": count}
            for aspect, count in aspect_counts.most_common(25)
        ],
        "dataset_path": str(dataset_path),
        "rows_scanned": scanned_rows,
        "scan_limited": max_rows is not None,
        "max_rows": max_rows,
    }
 
 
def compare_products(
    dataset_path: Path,
    product_names: list[str],
    filters: ReviewFilter,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    requested = [name.strip() for name in product_names if name and name.strip()]
    requested_lookup = {name.lower(): name for name in requested}
    by_product: dict[str, dict[str, Any]] = {
        canonical: {
            "product_name": canonical,
            "product_display_name_counts": Counter(),
            "product_source_url_counts": Counter(),
            "review_count": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
            "sentiment_sum": 0.0,
            "sentiment_count": 0,
            "sentiment_distribution": Counter(),
            "source_distribution": Counter(),
            "positive_aspects": Counter(),
            "negative_aspects": Counter(),
            "latest_review_date": None,
        }
        for canonical in requested
    }
 
    scanned_rows = 0
    for row in _iter_rows(dataset_path, max_rows=max_rows):
        scanned_rows += 1
        product_key = _normalized_text(row.get("product_name"))
        if product_key not in requested_lookup:
            continue
        if not _matches_filter(row, filters):
            continue
        product_name = requested_lookup[product_key]
        bucket = by_product[product_name]
        bucket["review_count"] += 1
 
        display_name = _clean_product_label_candidate(row.get("display_name"))
        if display_name:
            bucket["product_display_name_counts"][display_name] += 1
        source_url = str(row.get("source_url") or "").strip()
        if source_url:
            bucket["product_source_url_counts"][source_url] += 1
 
        rating = _coerce_float(row.get("rating_normalized"))
        if rating is not None:
            bucket["rating_sum"] += rating
            bucket["rating_count"] += 1
        score = _coerce_float(row.get("sentiment_score"))
        if score is not None:
            bucket["sentiment_sum"] += score
            bucket["sentiment_count"] += 1
 
        sentiment = str(row.get("sentiment_label") or "unknown").lower()
        bucket["sentiment_distribution"][sentiment] += 1
        bucket["source_distribution"][str(row.get("source") or "unknown").lower()] += 1
        aspect_target = (
            bucket["positive_aspects"]
            if sentiment == "positive"
            else bucket["negative_aspects"]
            if sentiment == "negative"
            else None
        )
        if aspect_target is not None:
            for aspect in _split_aspects(row.get("aspect_labels")):
                aspect_target[aspect] += 1
 
        parsed_date = _coerce_datetime(row.get("review_date"))
        if parsed_date is not None:
            latest = bucket["latest_review_date"]
            bucket["latest_review_date"] = parsed_date if latest is None else max(latest, parsed_date)
 
    products = []
    amazon_product_names = [
        bucket["product_name"]
        for bucket in by_product.values()
        if bucket["source_distribution"].most_common(1)
        and bucket["source_distribution"].most_common(1)[0][0] == "amazon"
        and _looks_like_machine_id(bucket["product_name"])
    ]
    amazon_titles = resolve_amazon_product_titles(amazon_product_names)
    for bucket in by_product.values():
        rating_count = bucket.pop("rating_count")
        rating_sum = bucket.pop("rating_sum")
        sentiment_count = bucket.pop("sentiment_count")
        sentiment_sum = bucket.pop("sentiment_sum")
        latest_review_date = bucket.pop("latest_review_date")
        product_display_name_counts = bucket.pop("product_display_name_counts")
        product_source_url_counts = bucket.pop("product_source_url_counts")
        dominant_source = bucket["source_distribution"].most_common(1)[0][0] if bucket["source_distribution"] else ""
        products.append(
            {
                **bucket,
                "product_title": amazon_titles.get(bucket["product_name"], ""),
                "product_label": _product_label_from_components(
                    source=dominant_source,
                    product_name=bucket["product_name"],
                    product_title=amazon_titles.get(bucket["product_name"]),
                    display_name_counts=product_display_name_counts,
                    source_url_counts=product_source_url_counts,
                ),
                "average_rating_normalized": round(rating_sum / rating_count, 4)
                if rating_count
                else None,
                "average_sentiment_score": round(sentiment_sum / sentiment_count, 4)
                if sentiment_count
                else None,
                "sentiment_distribution": [
                    {"sentiment_label": label, "count": count}
                    for label, count in sorted(bucket["sentiment_distribution"].items())
                ],
                "source_distribution": [
                    {"source": source, "count": count}
                    for source, count in sorted(bucket["source_distribution"].items())
                ],
                "top_positive_aspects": [
                    {"aspect": aspect, "count": count}
                    for aspect, count in bucket["positive_aspects"].most_common(5)
                ],
                "top_negative_aspects": [
                    {"aspect": aspect, "count": count}
                    for aspect, count in bucket["negative_aspects"].most_common(5)
                ],
                "latest_review_date": latest_review_date.isoformat() if latest_review_date else None,
            }
        )
 
    return {
        "products": products,
        "dataset_path": str(dataset_path),
        "rows_scanned": scanned_rows,
        "scan_limited": max_rows is not None,
        "max_rows": max_rows,
    }
 
 
def build_aspect_intelligence(
    dataset_path: Path,
    filters: ReviewFilter,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    aspect_counts: Counter[str] = Counter()
    aspect_sentiment: dict[str, Counter[str]] = defaultdict(Counter)
    aspect_score_sum: defaultdict[str, float] = defaultdict(float)
    aspect_score_count: Counter[str] = Counter()
    aspect_examples: dict[str, dict[str, str]] = {}
    rows_with_aspects = 0
    scanned_rows = 0
 
    for row in _iter_rows(dataset_path, max_rows=max_rows):
        scanned_rows += 1
        if not _matches_filter(row, filters):
            continue
        aspects = _split_aspects(row.get("aspect_labels"))
        if not aspects:
            continue
        rows_with_aspects += 1
        sentiment = str(row.get("sentiment_label") or "unknown").lower()
        score = _coerce_float(row.get("sentiment_score"))
        for aspect in aspects:
            aspect_counts[aspect] += 1
            aspect_sentiment[aspect][sentiment] += 1
            if score is not None:
                aspect_score_sum[aspect] += score
                aspect_score_count[aspect] += 1
            if aspect not in aspect_examples:
                aspect_examples[aspect] = {
                    "review_text": str(row.get("review_text") or "")[:280],
                    "source": str(row.get("source") or ""),
                    "product_name": str(row.get("product_name") or ""),
                }
 
    aspects = []
    for aspect, count in aspect_counts.most_common(50):
        score_count = aspect_score_count[aspect]
        sentiments = aspect_sentiment[aspect]
        aspects.append(
            {
                "aspect": aspect,
                "count": count,
                "positive_count": sentiments.get("positive", 0),
                "negative_count": sentiments.get("negative", 0),
                "neutral_count": sentiments.get("neutral", 0),
                "average_sentiment_score": round(aspect_score_sum[aspect] / score_count, 4)
                if score_count
                else None,
                "example": aspect_examples.get(aspect),
            }
        )
 
    return {
        "rows_with_aspects": rows_with_aspects,
        "aspects": aspects,
        "top_positive_aspects": sorted(
            aspects,
            key=lambda item: (item["positive_count"], item["count"]),
            reverse=True,
        )[:10],
        "top_negative_aspects": sorted(
            aspects,
            key=lambda item: (item["negative_count"], item["count"]),
            reverse=True,
        )[:10],
        "dataset_path": str(dataset_path),
        "rows_scanned": scanned_rows,
        "scan_limited": max_rows is not None,
        "max_rows": max_rows,
    }
 
 
def build_pipeline_status(
    *,
    data_dir: Path,
    logs_dir: Path,
    runtime_dir: Path,
    dags_dir: Path,
    normalized_path: Path,
    sentiment_path: Path,
    chroma_path: Path,
) -> dict[str, Any]:
    artifacts = []
    for label, path in (
        ("Normalized parquet", normalized_path),
        ("Sentiment parquet", sentiment_path),
        ("Vector store", sqlite_store_path(chroma_path)),
        ("HITL queue", runtime_dir / "hitl_queue.jsonl"),
        ("Embedding log", runtime_dir / "build_embeddings.log"),
    ):
        artifacts.append(
            {
                "name": label,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                if path.exists()
                else None,
            }
        )
 
    source_cards = []
    for source in SOURCE_NAMES:
        candidates = [
            data_dir / source,
            data_dir / f"{source}_reviews.jsonl",
            data_dir / "raw" / source,
            data_dir / "processed" / source,
        ]
        existing = [path for path in candidates if path.exists()]
        source_cards.append(
            {
                "source": source,
                "available": bool(existing),
                "paths": [str(path) for path in existing],
                "updated_at": max(
                    (
                        datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                        for path in existing
                    ),
                    default=None,
                ),
            }
        )
 
    dag_files = [
        {
            "name": path.name,
            "path": str(path),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        for path in sorted(dags_dir.glob("*.py"))
    ]
    log_files = [
        {
            "name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        for root in (logs_dir, runtime_dir)
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ][-25:]
 
    return {
        "artifacts": artifacts,
        "sources": source_cards,
        "dag_files": dag_files,
        "log_files": log_files,
        "vector_store": {
            "path": str(chroma_path),
            "sqlite_documents": sqlite_vector_count(chroma_path),
            "exists": chroma_path.exists(),
        },
    }
 
 