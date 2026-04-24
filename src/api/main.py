"""
ReviewPulse AI FastAPI backend.

Run:
    poetry run uvicorn src.api.main:app --reload --reload-dir src
"""

from __future__ import annotations

# ruff: noqa: E402

import importlib
import os
import re
import socket
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import chromadb
import pyarrow as pa
import pyarrow.dataset as ds
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.amazon_titles import resolve_amazon_product_titles
from src.app_logic import (
    ApiCache,
    annotate_and_filter_reviews,
    build_guardrail_answer,
    enqueue_hitl_request,
    evaluate_query,
    load_hitl_queue,
    make_cache_key,
)
from src.insights.arrow_metrics import build_arrow_data_insights
from src.insights.app_analytics import (
    ReviewFilter,
    build_aspect_intelligence,
    build_dashboard_summary,
    build_pipeline_status,
    build_sentiment_analytics,
    compare_products,
    explore_reviews,
    list_filter_options,
)
from src.insights import (
    build_normalization_explanations,
)
from src.retrieval.embedding_backend import (
    DEFAULT_EMBEDDING_MODEL,
    encode_texts,
    load_embedding_backend,
)
from src.retrieval.sqlite_vector_store import SQLiteReviewVectorStore, sqlite_store_exists


try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


load_dotenv(PROJECT_ROOT / ".env")
settings = get_settings()

CHROMA_DIR = str(settings.chroma_path)
PARQUET_PATH = str(settings.sentiment_parquet_path)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "reviewpulse_reviews")

app = FastAPI(title="ReviewPulse AI API", version="0.1.0")


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


ENABLE_GUARDRAILS = _bool_env("ENABLE_GUARDRAILS", True)
ENABLE_FAKE_DETECTION = _bool_env("ENABLE_FAKE_DETECTION", False)
ENABLE_HITL = _bool_env("ENABLE_HITL", False)
FAKE_REVIEW_THRESHOLD = _float_env("FAKE_REVIEW_THRESHOLD", 0.7)
CONFIDENCE_THRESHOLD = _float_env("CONFIDENCE_THRESHOLD", 0.5)
ANTHROPIC_TIMEOUT_SECONDS = _float_env("ANTHROPIC_TIMEOUT_SECONDS", 10.0)
HEALTHCHECK_TIMEOUT_SECONDS = _float_env("HEALTHCHECK_TIMEOUT_SECONDS", 3.0)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_TTL_SECONDS = _int_env("REDIS_TTL_SECONDS", 3600)
REDIS_VECTOR_CACHE_TTL_SECONDS = _int_env("REDIS_VECTOR_CACHE_TTL_SECONDS", 3600)
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
LOGS_DIR = PROJECT_ROOT / "logs"
DAGS_DIR = PROJECT_ROOT / "dags"


class SearchResponse(BaseModel):
    source: str
    product_name: str
    product_title: str | None = None
    product_label: str | None = None
    product_category: str
    display_name: str
    display_category: str
    entity_type: str
    aspect_labels: str = ""
    aspect_count: int = 0
    sentiment_label: str
    sentiment_score: float
    source_url: str
    distance: float
    review_text: str
    fake_review_score: float | None = None
    fake_review_label: str | None = None
    fake_review_flags: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    source: str
    product_title: str | None = None
    product_label: str | None = None
    display_name: str
    display_category: str
    entity_type: str
    source_url: str
    aspect_labels: str = ""
    aspect_count: int = 0
    sentiment_label: str
    sentiment_score: float
    fake_review_score: float | None = None
    fake_review_label: str | None = None
    fake_review_flags: list[str] = Field(default_factory=list)


class GuardrailPayload(BaseModel):
    allowed: bool
    action: str
    reason: str
    flags: list[str] = Field(default_factory=list)
    confidence: float


class HitlQueueResponse(BaseModel):
    request_id: str
    created_at: str
    status: str
    query: str
    source_filter: str | None = None
    n_results: int
    reason: str
    flags: list[str] = Field(default_factory=list)
    confidence: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    guardrail: GuardrailPayload | None = None
    filtered_review_count: int = 0
    hitl_request_id: str | None = None
    cache_hit: bool = False


class DependencyHealth(BaseModel):
    name: str
    status: str
    configured: bool
    latency_ms: float | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DetailedHealthResponse(BaseModel):
    status: str
    checked_at: str
    app: dict[str, Any]
    dependencies: dict[str, DependencyHealth]


class ReviewExplorerResponse(BaseModel):
    rows: list[dict[str, Any]]
    matched_count: int
    limit: int
    offset: int
    has_more: bool
    dataset_path: str


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ReviewPulse-API")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
        .getOrCreate()
    )


@lru_cache(maxsize=1)
def get_embedding_backend():
    return load_embedding_backend(
        chroma_path=settings.chroma_path,
        model_name=EMBEDDING_MODEL,
    )


def encode_query_text(query: str) -> list[float]:
    return encode_texts(
        get_embedding_backend(),
        [query],
        batch_size=1,
        show_progress_bar=False,
    )[0]


@lru_cache(maxsize=1)
def get_api_cache() -> ApiCache:
    return ApiCache(REDIS_URL)


def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def get_insights_dataset_path() -> Path | None:
    if settings.normalized_parquet_path.exists():
        return settings.normalized_parquet_path
    if settings.sentiment_parquet_path.exists():
        return settings.sentiment_parquet_path
    return None


def load_insights_rows() -> tuple[list[dict[str, Any]] | None, Path | None]:
    dataset_path = get_insights_dataset_path()
    if dataset_path is None:
        return None, None

    spark = get_spark()
    try:
        df = spark.read.parquet(str(dataset_path))
        rows = [row.asDict(recursive=True) for row in df.toLocalIterator()]
    finally:
        spark.stop()

    return rows, dataset_path


def _dataset_signature(dataset_path: Path) -> tuple[int, int, int]:
    if dataset_path.is_file():
        stat = dataset_path.stat()
        return (1, int(stat.st_size), int(stat.st_mtime_ns))

    parquet_files = [
        path
        for path in dataset_path.rglob("*.parquet")
        if path.is_file()
    ]
    if not parquet_files:
        return (0, 0, 0)

    total_size = 0
    latest_mtime = 0
    for path in parquet_files:
        stat = path.stat()
        total_size += int(stat.st_size)
        latest_mtime = max(latest_mtime, int(stat.st_mtime_ns))
    return (len(parquet_files), total_size, latest_mtime)


def parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _optional_float(value: float | None) -> float | None:
    return value if value is not None else None


def build_review_filter(
    *,
    query: str | None = None,
    source: str | None = None,
    sentiment: str | None = None,
    product: str | None = None,
    category: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
) -> ReviewFilter:
    return ReviewFilter(
        query=query.strip() if query and query.strip() else None,
        source=source.strip() if source and source.strip() else None,
        sentiment=sentiment.strip() if sentiment and sentiment.strip() else None,
        product=product.strip() if product and product.strip() else None,
        category=category.strip() if category and category.strip() else None,
        date_start=parse_optional_datetime(date_start),
        date_end=parse_optional_datetime(date_end),
        min_rating=_optional_float(min_rating),
        max_rating=_optional_float(max_rating),
    )


def build_source_stats_with_arrow(parquet_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    available_columns = set(dataset.schema.names)
    requested_columns = [
        column
        for column in ("source", "sentiment_label", "aspect_labels")
        if column in available_columns
    ]

    source_counts: Counter[str] = Counter()
    sentiment_counts: Counter[tuple[str, str]] = Counter()
    aspect_counts: Counter[str] = Counter()
    batch_size = int(os.getenv("API_STATS_BATCH_SIZE", "100000"))

    for record_batch in dataset.to_batches(
        columns=requested_columns,
        batch_size=max(1, batch_size),
    ):
        table = pa.Table.from_batches([record_batch])
        data = table.to_pydict()
        sources = [str(value or "") for value in data.get("source", [])]
        sentiments = [str(value or "") for value in data.get("sentiment_label", [])]
        aspects = [str(value or "") for value in data.get("aspect_labels", [])]

        source_counts.update(sources)
        sentiment_counts.update(zip(sources, sentiments))
        for aspect_value in aspects:
            for aspect in re.split(r"\s*,\s*", aspect_value):
                cleaned = aspect.strip()
                if cleaned:
                    aspect_counts[cleaned] += 1

    return {
        "source_counts": [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items())
        ],
        "sentiment_breakdown": [
            {"source": source, "sentiment_label": sentiment, "count": count}
            for (source, sentiment), count in sorted(sentiment_counts.items())
        ],
        "top_aspects": [
            {"aspect": aspect, "count": count}
            for aspect, count in sorted(
                aspect_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


@lru_cache(maxsize=4)
def build_source_stats_cached(
    parquet_path: str,
    dataset_signature: tuple[int, int, int],
) -> dict[str, list[dict[str, Any]]]:
    return build_source_stats_with_arrow(parquet_path)


@lru_cache(maxsize=4)
def build_data_insights_cached(
    parquet_path: str,
    dataset_signature: tuple[int, int, int],
) -> dict[str, Any]:
    return build_arrow_data_insights(Path(parquet_path))


def looks_like_machine_id(value: str) -> bool:
    if not value:
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{16,}", str(value).strip()))


def clean_display_name(meta: dict) -> str:
    source = str(meta.get("source", "")).strip()
    display_name = str(meta.get("display_name", "")).strip()
    product_name = str(meta.get("product_name", "")).strip()

    if source == "amazon":
        if display_name and not looks_like_machine_id(display_name.replace("Amazon Electronics Item ", "")):
            return display_name
        if display_name:
            return display_name
        if product_name and product_name.lower() != "unknown":
            return f"Amazon Electronics Item {product_name}"
        return "Amazon Product Review"

    if source == "yelp":
        if display_name and not looks_like_machine_id(display_name):
            return display_name
        return "Yelp Business Review"

    if source == "ebay":
        return display_name or product_name or "eBay Listing"

    if source == "ifixit":
        return display_name or product_name or "iFixit Guide"

    if source == "reddit":
        return display_name or "Reddit Discussion"

    if source == "youtube":
        return display_name or "YouTube Review"

    return display_name or product_name or "Unknown Item"


def clean_display_category(meta: dict) -> str:
    source = str(meta.get("source", "")).strip()
    display_category = str(meta.get("display_category", "")).strip()
    product_category = str(meta.get("product_category", "")).strip()

    if source == "amazon":
        return display_category or "Electronics Product"
    if source == "yelp":
        return display_category or "Local Business"
    if source == "ebay":
        return display_category or product_category or "Marketplace Listing"
    if source == "ifixit":
        return display_category or product_category or "Repair Guide"
    if source == "reddit":
        return display_category or product_category or "Forum Discussion"
    if source == "youtube":
        return display_category or "Video Review"
    return display_category or product_category or "Unknown Category"


def clean_entity_type(meta: dict) -> str:
    entity_type = str(meta.get("entity_type", "")).strip()
    if entity_type:
        return entity_type

    defaults = {
        "amazon": "product_review",
        "yelp": "business_review",
        "ebay": "listing_review",
        "ifixit": "repair_review",
        "reddit": "forum_post",
        "youtube": "video_transcript",
    }
    return defaults.get(str(meta.get("source", "")).strip(), "unknown")


def clean_product_label(meta: dict) -> str:
    source = str(meta.get("source", "")).strip().lower()
    product_name = str(meta.get("product_name", "")).strip()
    product_title = str(meta.get("product_title", "")).strip()

    if product_title:
        return product_title
    if product_name and not looks_like_machine_id(product_name):
        return product_name
    if source == "amazon" and product_name and product_name.lower() != "unknown":
        return f"Amazon Product (ASIN {product_name})"
    return product_name or "Unknown Product"


def attach_product_labels(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    amazon_titles = resolve_amazon_product_titles(
        str(item.get("product_name") or "")
        for item in items
        if str(item.get("source") or "").strip().lower() == "amazon"
        and looks_like_machine_id(str(item.get("product_name") or "").strip())
    )

    output: list[dict[str, Any]] = []
    for item in items:
        product_name = str(item.get("product_name") or "").strip()
        product_title = str(item.get("product_title") or "").strip()
        if not product_title and str(item.get("source") or "").strip().lower() == "amazon":
            product_title = str(amazon_titles.get(product_name) or "")

        enriched = {
            **item,
            "product_title": product_title or None,
        }
        enriched["product_label"] = clean_product_label(enriched)
        output.append(enriched)

    return output


def _retrieve_reviews_from_snowflake(query: str, source_filter: Optional[str] = None, n_results: int = 5) -> list[dict]:
    if not settings.snowflake_enabled:
        return []
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            password=settings.snowflake_password,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
            role=settings.snowflake_role or None,
        )
        # Keywords are restricted to word chars only — safe to embed directly
        keywords = [w for w in re.sub(r"[^\w\s]", "", query.lower()).split() if len(w) > 3][:5]
        if not keywords:
            keywords = [re.sub(r"[^\w\s]", "", query.lower()).strip() or "review"]
        like_parts = " OR ".join([f'"review_text" ILIKE \'%{kw}%\'' for kw in keywords])
        safe_source = re.sub(r"[^\w]", "", source_filter.lower()) if source_filter else ""
        source_clause = f" AND \"source\" ILIKE '{safe_source}'" if safe_source else ""
        sql = f"""
            SELECT "review_id", "review_text", "source", "product_name", "product_category",
                   "display_name", "display_category", "entity_type", "source_url", "review_date"
            FROM {settings.snowflake_normalized_table}
            WHERE ({like_parts}){source_clause}
            LIMIT {n_results}
        """
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0].lower() for desc in cur.description]
        cur.close()
        conn.close()
        results = []
        for row in rows[:n_results]:
            item = dict(zip(columns, row))
            results.append({
                "source": str(item.get("source", "")),
                "product_name": str(item.get("product_name", "") or ""),
                "product_category": str(item.get("product_category", "") or ""),
                "display_name": clean_display_name(item),
                "display_category": clean_display_category(item),
                "entity_type": clean_entity_type(item),
                "aspect_labels": "",
                "aspect_count": 0,
                "sentiment_label": "",
                "sentiment_score": 0.0,
                "source_url": str(item.get("source_url", "") or ""),
                "distance": 0.5,
                "review_text": str(item.get("review_text", "") or ""),
            })
        return attach_product_labels(results)
    except Exception:
        return []


def retrieve_reviews(query: str, source_filter: Optional[str] = None, n_results: int = 5):
    if not settings.chroma_path.exists():
        return _retrieve_reviews_from_snowflake(query, source_filter, n_results)

    try:
        embedding_backend = get_embedding_backend()
    except Exception:
        return []

    if sqlite_store_exists(settings.chroma_path):
        store = SQLiteReviewVectorStore(settings.chroma_path)
        try:
            results = store.search(
                query=query,
                embedding_backend=embedding_backend,
                source_filter=source_filter,
                n_results=n_results,
            )
        finally:
            store.close()
        return attach_product_labels([
            {
                **item,
                "display_name": clean_display_name(item),
                "display_category": clean_display_category(item),
                "entity_type": clean_entity_type(item),
            }
            for item in results
        ])

    try:
        collection = get_collection()
        query_embedding = encode_texts(
            embedding_backend,
            [query],
            batch_size=1,
            show_progress_bar=False,
        )[0]
    except Exception:
        return []

    where = {"source": source_filter.lower()} if source_filter else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        output.append(
            {
                "source": str(metadata.get("source", "")),
                "product_name": str(metadata.get("product_name", "")),
                "product_category": str(metadata.get("product_category", "")),
                "display_name": clean_display_name(metadata),
                "display_category": clean_display_category(metadata),
                "entity_type": clean_entity_type(metadata),
                "aspect_labels": str(metadata.get("aspect_labels", "")),
                "aspect_count": int(metadata.get("aspect_count", 0) or 0),
                "sentiment_label": str(metadata.get("sentiment_label", "")),
                "sentiment_score": float(metadata.get("sentiment_score", 0.0)),
                "source_url": str(metadata.get("source_url", "")),
                "distance": float(distance),
                "review_text": str(document),
            }
        )

    return attach_product_labels(output)


def apply_fake_review_filter(reviews: list[dict]) -> tuple[list[dict], int]:
    if not ENABLE_FAKE_DETECTION:
        return reviews, 0
    return annotate_and_filter_reviews(
        reviews,
        threshold=FAKE_REVIEW_THRESHOLD,
    )


def to_guardrail_payload(decision: Any) -> GuardrailPayload:
    return GuardrailPayload(**decision.to_dict())


def build_citations(retrieved: list[dict]) -> list[Citation]:
    return [
        Citation(
            source=item["source"],
            product_title=item.get("product_title"),
            product_label=item.get("product_label"),
            display_name=item["display_name"],
            display_category=item["display_category"],
            entity_type=item["entity_type"],
            source_url=item["source_url"],
            aspect_labels=item["aspect_labels"],
            aspect_count=item["aspect_count"],
            sentiment_label=item["sentiment_label"],
            sentiment_score=item["sentiment_score"],
            fake_review_score=item.get("fake_review_score"),
            fake_review_label=item.get("fake_review_label"),
            fake_review_flags=item.get("fake_review_flags") or [],
        )
        for item in retrieved
    ]


def _error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = type(exc).__name__
    return f"{type(exc).__name__}: {message}"[:500]


def _health_result(
    *,
    name: str,
    status: str,
    configured: bool,
    message: str,
    started_at: float | None = None,
    details: dict[str, Any] | None = None,
) -> DependencyHealth:
    latency_ms = None
    if started_at is not None:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return DependencyHealth(
        name=name,
        status=status,
        configured=configured,
        latency_ms=latency_ms,
        message=message,
        details=details or {},
    )


def check_s3_health() -> DependencyHealth:
    details = {
        "bucket": settings.s3_bucket_name,
        "region": settings.aws_region,
        "raw_prefix": settings.s3_raw_prefix,
        "processed_prefix": settings.s3_processed_prefix,
    }
    if not settings.s3_enabled:
        return _health_result(
            name="s3",
            status="skipped",
            configured=False,
            message="S3_BUCKET_NAME is not configured.",
            details=details,
        )

    started_at = time.perf_counter()
    try:
        boto3_module = importlib.import_module("boto3")
        config_module = importlib.import_module("botocore.config")
        client = boto3_module.client(
            "s3",
            region_name=settings.aws_region,
            config=config_module.Config(
                connect_timeout=HEALTHCHECK_TIMEOUT_SECONDS,
                read_timeout=HEALTHCHECK_TIMEOUT_SECONDS,
                retries={"max_attempts": 1},
            ),
        )
        client.head_bucket(Bucket=settings.s3_bucket_name)
        return _health_result(
            name="s3",
            status="healthy",
            configured=True,
            message="Connected to S3 and verified bucket access.",
            started_at=started_at,
            details=details,
        )
    except Exception as exc:
        return _health_result(
            name="s3",
            status="unhealthy",
            configured=True,
            message=_error_message(exc),
            started_at=started_at,
            details=details,
        )


def check_snowflake_health() -> DependencyHealth:
    details = {
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "warehouse": settings.snowflake_warehouse,
        "database": settings.snowflake_database,
        "schema": settings.snowflake_schema,
        "role": settings.snowflake_role,
    }
    if not settings.snowflake_enabled:
        return _health_result(
            name="snowflake",
            status="skipped",
            configured=False,
            message=(
                "Snowflake account, user, password, warehouse, database, "
                "and schema are not fully configured."
            ),
            details=details,
        )

    started_at = time.perf_counter()
    connection = None
    try:
        connector = importlib.import_module("snowflake.connector")
        connect_kwargs: dict[str, Any] = {
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "password": settings.snowflake_password,
            "warehouse": settings.snowflake_warehouse,
            "database": settings.snowflake_database,
            "schema": settings.snowflake_schema,
            "login_timeout": int(max(1, HEALTHCHECK_TIMEOUT_SECONDS)),
            "network_timeout": int(max(1, HEALTHCHECK_TIMEOUT_SECONDS)),
        }
        if settings.snowflake_role:
            connect_kwargs["role"] = settings.snowflake_role
        connection = connector.connect(**connect_kwargs)
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT CURRENT_VERSION(), CURRENT_WAREHOUSE(), "
                "CURRENT_DATABASE(), CURRENT_SCHEMA()"
            )
            row = cursor.fetchone()
        finally:
            cursor.close()

        if row:
            details = {
                **details,
                "version": str(row[0]),
                "current_warehouse": str(row[1]),
                "current_database": str(row[2]),
                "current_schema": str(row[3]),
            }
        return _health_result(
            name="snowflake",
            status="healthy",
            configured=True,
            message="Connected to Snowflake and executed a metadata query.",
            started_at=started_at,
            details=details,
        )
    except Exception as exc:
        return _health_result(
            name="snowflake",
            status="unhealthy",
            configured=True,
            message=_error_message(exc),
            started_at=started_at,
            details=details,
        )
    finally:
        if connection is not None:
            close = getattr(connection, "close", None)
            if callable(close):
                close()


def check_redis_health() -> DependencyHealth:
    parsed = urlparse(REDIS_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    details = {
        "host": host,
        "port": port,
        "database": parsed.path.lstrip("/") if parsed.path else "",
    }
    if not REDIS_URL:
        return _health_result(
            name="redis",
            status="skipped",
            configured=False,
            message="REDIS_URL is not configured.",
            details=details,
        )

    started_at = time.perf_counter()
    try:
        with socket.create_connection(
            (host, port),
            timeout=HEALTHCHECK_TIMEOUT_SECONDS,
        ) as connection:
            connection.settimeout(HEALTHCHECK_TIMEOUT_SECONDS)
            connection.sendall(b"*1\r\n$4\r\nPING\r\n")
            response = connection.recv(256).decode("utf-8", errors="replace").strip()
        if response.startswith("+PONG"):
            status = "healthy"
            message = "Connected to Redis and received PONG."
        elif response.startswith("-NOAUTH"):
            status = "degraded"
            message = "Connected to Redis, but authentication is required for PING."
        else:
            status = "unhealthy"
            message = f"Connected to Redis, but PING returned: {response[:120]}"
        return _health_result(
            name="redis",
            status=status,
            configured=True,
            message=message,
            started_at=started_at,
            details=details,
        )
    except Exception as exc:
        return _health_result(
            name="redis",
            status="unhealthy",
            configured=True,
            message=_error_message(exc),
            started_at=started_at,
            details=details,
        )


def build_detailed_health() -> DetailedHealthResponse:
    dependencies = {
        "s3": check_s3_health(),
        "snowflake": check_snowflake_health(),
        "redis": check_redis_health(),
    }
    statuses = {dependency.status for dependency in dependencies.values()}
    if "unhealthy" in statuses:
        status = "unhealthy"
    elif "degraded" in statuses:
        status = "degraded"
    else:
        status = "healthy"

    return DetailedHealthResponse(
        status=status,
        checked_at=datetime.now(UTC).isoformat(),
        app={
            "name": settings.app_name,
            "environment": settings.app_env,
            "guardrails_enabled": ENABLE_GUARDRAILS,
            "fake_detection_enabled": ENABLE_FAKE_DETECTION,
            "hitl_enabled": ENABLE_HITL,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "healthcheck_timeout_seconds": HEALTHCHECK_TIMEOUT_SECONDS,
        },
        dependencies=dependencies,
    )


def fallback_answer(query: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "I could not find relevant reviews for that query."

    positives: list[str] = []
    negatives: list[str] = []
    neutrals: list[str] = []

    for item in retrieved[:5]:
        summary_line = f"{item['display_name']} ({item['display_category']}, {item['source']})"
        if item["sentiment_label"] == "positive":
            positives.append(summary_line)
        elif item["sentiment_label"] == "negative":
            negatives.append(summary_line)
        else:
            neutrals.append(summary_line)

    parts = [f"Based on the retrieved reviews for '{query}', here is a grounded summary."]
    if positives:
        parts.append("Positive evidence appears in reviews such as " + ", ".join(positives[:2]) + ".")
    if negatives:
        parts.append("Negative evidence appears in reviews such as " + ", ".join(negatives[:2]) + ".")
    if neutrals and not negatives:
        parts.append("Some results are neutral or mixed, including " + ", ".join(neutrals[:2]) + ".")
    parts.append("This answer is based only on the retrieved review text and citations below.")
    return " ".join(parts)


def generate_grounded_answer(query: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "I could not find relevant reviews for that query."

    if not (ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY):
        return fallback_answer(query, retrieved)

    try:
        client = Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=ANTHROPIC_TIMEOUT_SECONDS,
        )

        context_blocks = []
        for index, item in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[Review {index}]\n"
                f"Source: {item['source']}\n"
                f"Display Name: {item['display_name']}\n"
                f"Display Category: {item['display_category']}\n"
                f"Entity Type: {item['entity_type']}\n"
                f"Aspects: {item.get('aspect_labels', '')}\n"
                f"Sentiment: {item['sentiment_label']} ({item['sentiment_score']})\n"
                f"URL: {item['source_url']}\n"
                f"Text: {item['review_text']}\n"
            )

        prompt = f"""
You are answering a user question ONLY from the retrieved reviews below.

Rules:
- Use only the provided reviews.
- Do not invent product facts.
- If evidence is weak or mixed, say so clearly.
- Keep the answer concise and useful.
- Use the display name and display category naturally when referring to retrieved items.
- Do not claim certainty when the evidence is limited.

User query:
{query}

Retrieved reviews:
{chr(10).join(context_blocks)}

Now answer the user query using only this evidence.
"""

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )

        parts = response.content
        if parts and hasattr(parts[0], "text"):
            return parts[0].text.strip()
        return fallback_answer(query, retrieved)
    except Exception as error:
        print(f"Anthropic call failed: {error}")
        return fallback_answer(query, retrieved)


@app.get("/health")
def health():
    cache = get_api_cache()
    return {
        "status": "ok",
        "detailed_health_url": "/health/detailed",
        "guardrails_enabled": ENABLE_GUARDRAILS,
        "fake_detection_enabled": ENABLE_FAKE_DETECTION,
        "hitl_enabled": ENABLE_HITL,
        "cache_enabled": cache.enabled,
        "cache_disabled_reason": cache.disabled_reason,
    }


@app.get("/health/detailed", response_model=DetailedHealthResponse)
def detailed_health():
    return build_detailed_health()


@app.get("/dashboard/summary")
def dashboard_summary():
    dataset_path = get_insights_dataset_path()
    if dataset_path is None:
        return {"error": "No parquet dataset found. Run the normalization pipeline first."}

    return build_dashboard_summary(
        normalized_path=settings.normalized_parquet_path,
        sentiment_path=settings.sentiment_parquet_path,
        chroma_path=settings.chroma_path,
        hitl_queue_path=RUNTIME_DIR / "hitl_queue.jsonl",
        data_dir=settings.data_dir,
    )


@app.get("/reviews/options")
def review_filter_options(
    limit: int = Query(200, ge=1, le=1000),
    max_rows: Optional[int] = Query(None, ge=1, le=1000000),
):
    dataset_path = settings.sentiment_parquet_path if settings.sentiment_parquet_path.exists() else get_insights_dataset_path()
    if dataset_path is None:
        return {"error": "No parquet dataset found. Run the normalization pipeline first."}
    return list_filter_options(dataset_path, limit=limit, max_rows=max_rows)


@app.get("/reviews/explore", response_model=ReviewExplorerResponse | dict[str, str])
def review_explorer(
    query: Optional[str] = Query(None, description="Text filter over review/product/aspects"),
    source: Optional[str] = Query(None, description="Source filter"),
    sentiment: Optional[str] = Query(None, description="Sentiment label filter"),
    product: Optional[str] = Query(None, description="Exact product_name filter"),
    category: Optional[str] = Query(None, description="Category substring filter"),
    date_start: Optional[str] = Query(None, description="ISO start date"),
    date_end: Optional[str] = Query(None, description="ISO end date"),
    min_rating: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_rating: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    dataset_path = settings.sentiment_parquet_path if settings.sentiment_parquet_path.exists() else get_insights_dataset_path()
    if dataset_path is None:
        return {"error": "No parquet dataset found. Run the normalization pipeline first."}
    filters = build_review_filter(
        query=query,
        source=source,
        sentiment=sentiment,
        product=product,
        category=category,
        date_start=date_start,
        date_end=date_end,
        min_rating=min_rating,
        max_rating=max_rating,
    )
    return explore_reviews(dataset_path, filters, limit=limit, offset=offset)


@app.get("/stats/sources")
def source_stats():
    parquet_path = settings.sentiment_parquet_path
    if not parquet_path.exists():
        return {"error": "Sentiment parquet not found. Run the earlier pipeline steps first."}

    return build_source_stats_cached(
        str(parquet_path),
        _dataset_signature(parquet_path),
    )


@app.get("/data/profile")
def data_profile():
    dataset_path = get_insights_dataset_path()
    if dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to create parquet data for profiling."
            )
        }

    insights = build_data_insights_cached(
        str(dataset_path),
        _dataset_signature(dataset_path),
    )
    return insights["profile"]


@app.get("/data/compare")
def data_compare():
    dataset_path = get_insights_dataset_path()
    if dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to compare sources."
            )
        }

    insights = build_data_insights_cached(
        str(dataset_path),
        _dataset_signature(dataset_path),
    )
    return insights["comparison"]


@app.get("/data/quality")
def data_quality():
    dataset_path = get_insights_dataset_path()
    if dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to compute data quality metrics."
            )
        }

    insights = build_data_insights_cached(
        str(dataset_path),
        _dataset_signature(dataset_path),
    )
    return insights["quality"]


@app.get("/analytics/sentiment")
def sentiment_analytics(
    query: Optional[str] = None,
    source: Optional[str] = None,
    sentiment: Optional[str] = None,
    product: Optional[str] = None,
    category: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    min_rating: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_rating: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_rows: Optional[int] = Query(None, ge=1, le=1000000),
):
    dataset_path = settings.sentiment_parquet_path
    if not dataset_path.exists():
        return {"error": "Sentiment parquet not found. Run the sentiment pipeline first."}
    filters = build_review_filter(
        query=query,
        source=source,
        sentiment=sentiment,
        product=product,
        category=category,
        date_start=date_start,
        date_end=date_end,
        min_rating=min_rating,
        max_rating=max_rating,
    )
    return build_sentiment_analytics(dataset_path, filters, max_rows=max_rows)


@app.get("/products/compare")
def product_compare(
    products: str = Query(..., description="Comma-separated product_name values"),
    source: Optional[str] = None,
    category: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    max_rows: Optional[int] = Query(None, ge=1, le=1000000),
):
    dataset_path = settings.sentiment_parquet_path
    if not dataset_path.exists():
        return {"error": "Sentiment parquet not found. Run the sentiment pipeline first."}
    product_names = [part.strip() for part in products.split(",") if part.strip()]
    if not product_names:
        return {"error": "Provide at least one product name."}
    filters = build_review_filter(
        source=source,
        category=category,
        date_start=date_start,
        date_end=date_end,
    )
    return compare_products(dataset_path, product_names, filters, max_rows=max_rows)


@app.get("/aspects/summary")
def aspect_summary(
    query: Optional[str] = None,
    source: Optional[str] = None,
    sentiment: Optional[str] = None,
    product: Optional[str] = None,
    category: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    max_rows: Optional[int] = Query(None, ge=1, le=1000000),
):
    dataset_path = settings.sentiment_parquet_path
    if not dataset_path.exists():
        return {"error": "Sentiment parquet not found. Run the sentiment pipeline first."}
    filters = build_review_filter(
        query=query,
        source=source,
        sentiment=sentiment,
        product=product,
        category=category,
        date_start=date_start,
        date_end=date_end,
    )
    return build_aspect_intelligence(dataset_path, filters, max_rows=max_rows)


@app.get("/pipeline/status")
def pipeline_status():
    return build_pipeline_status(
        data_dir=settings.data_dir,
        logs_dir=LOGS_DIR,
        runtime_dir=RUNTIME_DIR,
        dags_dir=DAGS_DIR,
        normalized_path=settings.normalized_parquet_path,
        sentiment_path=settings.sentiment_parquet_path,
        chroma_path=settings.chroma_path,
    )


@app.get("/data/normalize/explain")
def data_normalize_explain(
    source: Optional[str] = Query(
        None,
        description="Optional source filter: amazon/yelp/ebay/ifixit/youtube",
    ),
    sample_index: int = Query(0, ge=0, le=100),
):
    return build_normalization_explanations(source=source, sample_index=sample_index)


@app.get("/search/semantic", response_model=list[SearchResponse])
def semantic_search(
    query: str = Query(..., description="Natural-language query"),
    source_filter: Optional[str] = Query(
        None,
        description="amazon/yelp/ebay/ifixit/reddit/youtube",
    ),
    n_results: int = Query(5, ge=1, le=20),
):
    cache_key = make_cache_key(
        "semantic_search",
        query=query,
        source_filter=source_filter,
        n_results=n_results,
        fake_detection_enabled=ENABLE_FAKE_DETECTION,
        fake_review_threshold=FAKE_REVIEW_THRESHOLD,
    )
    cache = get_api_cache()
    cached = cache.get_json(cache_key)
    if isinstance(cached, list):
        return [SearchResponse(**item) for item in cached]

    results = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    results, _filtered_count = apply_fake_review_filter(results)
    response = [SearchResponse(**item) for item in results]
    cache.set_json(
        cache_key,
        [item.model_dump() for item in response],
        ttl_seconds=REDIS_VECTOR_CACHE_TTL_SECONDS,
    )
    return response


@app.get("/hitl/queue", response_model=list[HitlQueueResponse])
def hitl_queue(
    status: Optional[str] = Query("pending", description="Queue status filter"),
    limit: int = Query(100, ge=1, le=500),
):
    rows = load_hitl_queue(status=status or None, limit=limit)
    return [HitlQueueResponse(**row) for row in rows]


@app.get("/chat", response_model=ChatResponse)
def chat(
    query: str = Query(..., description="User question"),
    source_filter: Optional[str] = Query(None, description="Optional source filter"),
    n_results: int = Query(5, ge=1, le=10),
):
    decision = None
    if ENABLE_GUARDRAILS:
        decision = evaluate_query(
            query,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            hitl_enabled=ENABLE_HITL,
        )
        if not decision.allowed:
            queued = None
            if decision.action == "needs_human_review":
                queued = enqueue_hitl_request(
                    query=query,
                    decision=decision,
                    source_filter=source_filter,
                    n_results=n_results,
                )

            return ChatResponse(
                answer=build_guardrail_answer(decision),
                citations=[],
                guardrail=to_guardrail_payload(decision),
                hitl_request_id=queued.request_id if queued else None,
            )

    cache_key = make_cache_key(
        "chat",
        query=query,
        source_filter=source_filter,
        n_results=n_results,
        guardrails_enabled=ENABLE_GUARDRAILS,
        fake_detection_enabled=ENABLE_FAKE_DETECTION,
        fake_review_threshold=FAKE_REVIEW_THRESHOLD,
        anthropic_model=ANTHROPIC_MODEL if ANTHROPIC_API_KEY else "fallback",
    )
    cache = get_api_cache()
    cached = cache.get_json(cache_key)
    if isinstance(cached, dict):
        cached["cache_hit"] = True
        return ChatResponse(**cached)

    retrieved = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    retrieved, filtered_count = apply_fake_review_filter(retrieved)
    answer = generate_grounded_answer(query, retrieved)

    response = ChatResponse(
        answer=answer,
        citations=build_citations(retrieved),
        guardrail=to_guardrail_payload(decision) if decision else None,
        filtered_review_count=filtered_count,
    )
    cache.set_json(cache_key, response.model_dump(), ttl_seconds=REDIS_TTL_SECONDS)
    return response
