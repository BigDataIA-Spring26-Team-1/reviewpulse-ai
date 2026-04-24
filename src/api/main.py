"""
ReviewPulse AI FastAPI backend.

Run:
    poetry run uvicorn src.api.main:app --reload --reload-dir src
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import re
import sys
from collections import Counter
<<<<<<< HEAD
from functools import lru_cache
=======
>>>>>>> 2feac8497b1b0ebd59798af410a2cbb616d914dc
from pathlib import Path
from typing import Any, Optional

import chromadb
import pyarrow as pa
import pyarrow.dataset as ds
from fastapi import FastAPI, Query
from pydantic import BaseModel
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
from src.insights import (
    build_dataset_profile,
    build_normalization_explanations,
    build_quality_metrics,
    build_source_comparison,
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
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "reviewpulse_reviews")

app = FastAPI(title="ReviewPulse AI API", version="0.1.0")


class SearchResponse(BaseModel):
    source: str
    product_name: str
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


class Citation(BaseModel):
    source: str
    display_name: str
    display_category: str
    entity_type: str
    source_url: str
    aspect_labels: str = ""
    aspect_count: int = 0
    sentiment_label: str
    sentiment_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ReviewPulse-API")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
        .getOrCreate()
    )


<<<<<<< HEAD
@lru_cache(maxsize=1)
=======
>>>>>>> 2feac8497b1b0ebd59798af410a2cbb616d914dc
def get_embedding_backend():
    return load_embedding_backend(
        chroma_path=settings.chroma_path,
        model_name=EMBEDDING_MODEL,
    )
<<<<<<< HEAD


def encode_query_text(query: str) -> list[float]:
    return encode_texts(
        get_embedding_backend(),
        [query],
        batch_size=1,
        show_progress_bar=False,
    )[0]
=======
>>>>>>> 2feac8497b1b0ebd59798af410a2cbb616d914dc


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
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


def retrieve_reviews(query: str, source_filter: Optional[str] = None, n_results: int = 5):
    if not os.path.exists(CHROMA_DIR):
        return []

<<<<<<< HEAD
    try:
        embedding_backend = get_embedding_backend()
    except Exception:
        return []

=======
    embedding_backend = get_embedding_backend()
>>>>>>> 2feac8497b1b0ebd59798af410a2cbb616d914dc
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
        return [
            {
                **item,
                "display_name": clean_display_name(item),
                "display_category": clean_display_category(item),
                "entity_type": clean_entity_type(item),
            }
            for item in results
        ]

<<<<<<< HEAD
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
=======
    collection = get_collection()
    query_embedding = encode_texts(embedding_backend, [query])[0]
>>>>>>> 2feac8497b1b0ebd59798af410a2cbb616d914dc

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

    return output


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
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

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
    return {"status": "ok"}


@app.get("/stats/sources")
def source_stats():
    if not os.path.exists(PARQUET_PATH):
        return {"error": "Sentiment parquet not found. Run the earlier pipeline steps first."}

    return build_source_stats_with_arrow(PARQUET_PATH)


@app.get("/data/profile")
def data_profile():
    rows, dataset_path = load_insights_rows()
    if rows is None or dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to create parquet data for profiling."
            )
        }

    return build_dataset_profile(rows, dataset_path=str(dataset_path))


@app.get("/data/compare")
def data_compare():
    rows, dataset_path = load_insights_rows()
    if rows is None or dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to compare sources."
            )
        }

    comparison = build_source_comparison(rows)
    comparison["dataset_path"] = str(dataset_path)
    return comparison


@app.get("/data/quality")
def data_quality():
    rows, dataset_path = load_insights_rows()
    if rows is None or dataset_path is None:
        return {
            "error": (
                "No normalized dataset found. Run the normalization pipeline first "
                "to compute data quality metrics."
            )
        }

    quality = build_quality_metrics(rows)
    quality["dataset_path"] = str(dataset_path)
    return quality


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
    results = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    return [SearchResponse(**item) for item in results]


@app.get("/chat", response_model=ChatResponse)
def chat(
    query: str = Query(..., description="User question"),
    source_filter: Optional[str] = Query(None, description="Optional source filter"),
    n_results: int = Query(5, ge=1, le=10),
):
    retrieved = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    answer = generate_grounded_answer(query, retrieved)

    citations = [
        Citation(
            source=item["source"],
            display_name=item["display_name"],
            display_category=item["display_category"],
            entity_type=item["entity_type"],
            source_url=item["source_url"],
            aspect_labels=item["aspect_labels"],
            aspect_count=item["aspect_count"],
            sentiment_label=item["sentiment_label"],
            sentiment_score=item["sentiment_score"],
        )
        for item in retrieved
    ]

    return ChatResponse(answer=answer, citations=citations)
