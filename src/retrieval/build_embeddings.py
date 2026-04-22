"""
ReviewPulse AI embedding build stage.

Run:
    poetry run python src/retrieval/build_embeddings.py
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.common.run_context import build_run_context
from src.common.spark_runtime import ensure_local_hadoop_home


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
    hadoop_home = ensure_local_hadoop_home(PROJECT_ROOT)
    builder = (
        SparkSession.builder
        .appName("ReviewPulse-BuildEmbeddings")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
    )
    if hadoop_home is not None:
        builder = builder.config("spark.hadoop.hadoop.home.dir", str(hadoop_home))
    return builder.getOrCreate()


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("retrieval.build_embeddings")
    run_context = build_run_context(stage="build_embeddings")
    started_at = time.perf_counter()
    storage_manager = S3StorageManager.from_settings(settings)

    log_event(
        logger,
        "pipeline_run_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        status="started",
    )

    if not settings.sentiment_parquet_path.exists():
        log_event(
            logger,
            "embeddings_generation_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(settings.sentiment_parquet_path),
            status="failed",
            error_type="MissingInputError",
            error_message="Sentiment parquet not found. Run sentiment scoring first.",
        )
        raise RuntimeError("Sentiment parquet not found. Run sentiment scoring first.")

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
        log_event(
            logger,
            "embeddings_generation_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="failed",
            error_type="EmptyDatasetError",
            error_message="No rows found for embedding.",
        )
        raise RuntimeError("No rows found for embedding.")

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

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    SentenceTransformer = _require_sentence_transformer()
    model = SentenceTransformer(model_name)

    log_event(
        logger,
        "embeddings_generation_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=len(documents),
        status="started",
    )
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True)

    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)

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
    run_prefix = storage_manager.resolver.processed_run_prefix("chromadb", run_context.run_id)
    current_prefix = storage_manager.resolver.processed_current_prefix("chromadb")

    log_event(
        logger,
        "s3_upload_started",
        stage="chromadb",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(settings.chroma_path),
        output_path=run_prefix,
        status="started",
    )
    uploaded = storage_manager.upload_directory(settings.chroma_path, run_prefix)
    log_event(
        logger,
        "s3_upload_completed",
        stage="chromadb",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(settings.chroma_path),
        output_path=run_prefix,
        file_count=len(uploaded),
        status="success",
    )
    promotion = storage_manager.promote_run_prefix(
        run_prefix,
        current_prefix,
        run_id=run_context.run_id,
        metadata={"stage": "chromadb", "status": "success"},
    )
    log_event(
        logger,
        "latest_run_promoted",
        stage="chromadb",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        output_path=current_prefix,
        file_count=promotion["copied_count"],
        status="success",
    )
    log_event(
        logger,
        "embeddings_generation_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=len(documents),
        output_path=str(settings.chroma_path),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
    )


if __name__ == "__main__":
    main()
