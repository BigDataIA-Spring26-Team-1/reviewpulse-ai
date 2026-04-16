"""
ReviewPulse AI embedding build stage.

Run:
    poetry run python src/retrieval/build_embeddings.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings


DEPENDENCY_SETUP_HINT = (
    "Install project dependencies with `poetry install` and use the Poetry environment "
    "before running the embeddings pipeline."
)


def _require_pyspark():
    try:
        pyspark_sql = importlib.import_module("pyspark.sql")
        SparkSession = pyspark_sql.SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `pyspark`. " + DEPENDENCY_SETUP_HINT
        ) from exc
    return SparkSession


def _require_sentence_transformer():
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
        SentenceTransformer = sentence_transformers.SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `sentence-transformers`. " + DEPENDENCY_SETUP_HINT
        ) from exc
    return SentenceTransformer


def _require_chromadb():
    try:
        chromadb = importlib.import_module("chromadb")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `chromadb`. " + DEPENDENCY_SETUP_HINT
        ) from exc
    return chromadb


def build_spark() -> Any:
    settings = get_settings()
    SparkSession = _require_pyspark()
    return (
        SparkSession.builder
        .appName("ReviewPulse-BuildEmbeddings")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
        .getOrCreate()
    )


def main() -> None:
    settings = get_settings()

    print("=" * 60)
    print("REVIEWPULSE AI BUILD EMBEDDINGS")
    print("=" * 60)

    if not settings.sentiment_parquet_path.exists():
        print(f"Input parquet not found: {settings.sentiment_parquet_path}")
        print("Run sentiment scoring first.")
        return

    spark = build_spark()
    df = spark.read.parquet(str(settings.sentiment_parquet_path))
    df = df.filter(df.review_text.isNotNull())

    amazon_df = df.filter(df.source == "amazon").limit(2000)
    yelp_df = df.filter(df.source == "yelp").limit(2000)
    ebay_df = df.filter(df.source == "ebay").limit(1000)
    ifixit_df = df.filter(df.source == "ifixit").limit(1000)
    youtube_df = df.filter(df.source == "youtube").limit(250)
    reddit_df = df.filter(df.source == "reddit").limit(500)

    df = amazon_df.unionByName(yelp_df)
    for source_df in [ebay_df, ifixit_df, youtube_df, reddit_df]:
        df = df.unionByName(source_df)

    print("\nEmbedding subset counts by source:")
    df.groupBy("source").count().orderBy("source").show(truncate=False)

    rows = df.select(
        "review_id",
        "review_text",
        "source",
        "product_name",
        "product_category",
        "display_name",
        "display_category",
        "entity_type",
        "sentiment_label",
        "sentiment_score",
        "review_date",
        "source_url",
    ).collect()

    spark.stop()

    if not rows:
        print("No rows found for embedding.")
        return

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, object]] = []

    for row in rows:
        review_text = row["review_text"]
        if not review_text or len(review_text.strip()) < 20:
            continue

        ids.append(str(row["review_id"]))
        documents.append(review_text)
        metadatas.append(
            {
                "source": str(row["source"] or ""),
                "product_name": str(row["product_name"] or ""),
                "product_category": str(row["product_category"] or ""),
                "display_name": str(row["display_name"] or ""),
                "display_category": str(row["display_category"] or ""),
                "entity_type": str(row["entity_type"] or ""),
                "sentiment_label": str(row["sentiment_label"] or ""),
                "sentiment_score": float(row["sentiment_score"] or 0.0),
                "review_date": str(row["review_date"] or ""),
                "source_url": str(row["source_url"] or ""),
            }
        )

    print(f"Documents to embed: {len(documents)}")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {model_name}")
    SentenceTransformer = _require_sentence_transformer()
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True)

    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)

    print("Creating ChromaDB collection...")
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION_NAME", "reviewpulse_reviews")
    )

    batch_size = 500
    for index in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[index:index + batch_size],
            documents=documents[index:index + batch_size],
            embeddings=embeddings[index:index + batch_size].tolist(),
            metadatas=metadatas[index:index + batch_size],
        )
        print(f"Loaded batch {index} to {min(index + batch_size, len(documents))}")

    print("\nDone.")
    print(f"ChromaDB stored at: {settings.chroma_path}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
