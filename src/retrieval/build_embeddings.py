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

import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.common.run_context import build_run_context
from src.common.spark_runtime import ensure_local_hadoop_home
from src.retrieval.embedding_backend import (
    encode_texts,
    load_embedding_backend,
    write_embedding_backend_metadata,
)


DEPENDENCY_SETUP_HINT = (
    "Install project dependencies with `poetry install` and use the Poetry environment "
    "before running the embeddings pipeline."
)

EMBEDDING_COLUMNS = (
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
)

WINDOWS_NATIVE_PARQUET_ERROR_MARKERS = (
    "NativeIO$Windows.access0",
    "UnsatisfiedLinkError",
    "Access denied",
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


def _is_windows_native_parquet_error(exc: BaseException) -> bool:
    if os.name != "nt":
        return False

    current: BaseException | None = exc
    while current is not None:
        message = f"{type(current).__name__}: {current}"
        if "NativeIO$Windows.access0" in message and "Access denied" in message:
            return True
        if all(marker in message for marker in WINDOWS_NATIVE_PARQUET_ERROR_MARKERS[:2]):
            return True
        current = current.__cause__ or current.__context__
    return False


def _prepare_embedding_dataframe(df: Any) -> Any:
    return df.filter("review_text IS NOT NULL").select(*EMBEDDING_COLUMNS)


def _load_embedding_rows_with_arrow(input_path: Path) -> list[dict[str, object]]:
    dataset = ds.dataset(str(input_path), format="parquet")
    available_columns = [column for column in EMBEDDING_COLUMNS if column in dataset.schema.names]
    rows: list[dict[str, object]] = []
    for batch in dataset.to_batches(columns=available_columns):
        for row in batch.to_pylist():
            rows.append({column: row.get(column) for column in EMBEDDING_COLUMNS})
    return rows


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

    spark = None
    try:
        spark = build_spark()
        df = spark.read.parquet(str(settings.sentiment_parquet_path))
        rows = _prepare_embedding_dataframe(df).collect()
    except Exception as exc:
        if spark is not None:
            spark.stop()
            spark = None
        if not _is_windows_native_parquet_error(exc):
            raise
        log_event(
            logger,
            "parquet_read_fallback",
            level=logging.WARNING,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(settings.sentiment_parquet_path),
            status="fallback",
            error_type=type(exc).__name__,
            error_message="Spark parquet read failed on Windows; falling back to pyarrow.",
        )
        rows = _load_embedding_rows_with_arrow(settings.sentiment_parquet_path)
    finally:
        if spark is not None:
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
    backend = load_embedding_backend(model_name=model_name)
    if backend.fallback_reason:
        log_event(
            logger,
            "embedding_backend_fallback",
            level=logging.WARNING,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="fallback",
            error_type="EmbeddingBackendFallback",
            error_message=backend.fallback_reason,
            output_path=backend.model_name,
        )

    log_event(
        logger,
        "embeddings_generation_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=len(documents),
        entity_type=backend.backend_name,
        output_path=backend.model_name,
        status="started",
    )
    embeddings = encode_texts(backend, documents, batch_size=64, show_progress_bar=True)

    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
    write_embedding_backend_metadata(settings.chroma_path, backend)

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
            embeddings=embeddings[index:index + batch_size],
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
