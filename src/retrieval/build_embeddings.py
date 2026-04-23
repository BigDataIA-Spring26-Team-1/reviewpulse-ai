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

from src.common.run_context import build_run_context
from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.retrieval.embedding_backend import (
    DEFAULT_EMBEDDING_MODEL,
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
    "aspect_labels",
    "aspect_count",
    "sentiment_label",
    "sentiment_score",
    "review_date",
    "source_url",
)


def _require_chromadb():
    try:
        chromadb = importlib.import_module("chromadb")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `chromadb`. " + DEPENDENCY_SETUP_HINT
        ) from exc
    return chromadb


def _prepare_embedding_dataframe(dataframe: Any) -> Any:
    return dataframe.filter("review_text IS NOT NULL").select(*EMBEDDING_COLUMNS)


def _load_embedding_rows_with_arrow(input_path: Path) -> list[dict[str, object]]:
    dataset = ds.dataset(str(input_path), format="parquet")
    available_columns = set(dataset.schema.names)
    requested_columns = [column for column in EMBEDDING_COLUMNS if column in available_columns]
    table = dataset.to_table(columns=requested_columns)

    rows: list[dict[str, object]] = []
    for raw_row in table.to_pylist():
        row = dict(raw_row)
        for column in EMBEDDING_COLUMNS:
            if column in row:
                continue
            if column == "aspect_count":
                row[column] = 0
            elif column == "sentiment_score":
                row[column] = 0.0
            else:
                row[column] = ""
        rows.append(row)
    return rows


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("retrieval.build_embeddings")
    run_context = build_run_context(stage="build_embeddings")
    started_at = time.perf_counter()
    storage_manager = (
        S3StorageManager.from_settings(settings)
        if settings.s3_enabled
        else None
    )

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

    rows = _load_embedding_rows_with_arrow(settings.sentiment_parquet_path)
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
        review_text = str(row.get("review_text", "") or "")
        if len(review_text.strip()) < 20:
            continue

        ids.append(str(row.get("review_id", "")))
        documents.append(review_text)
        metadatas.append(
            {
                "source": str(row.get("source", "") or ""),
                "product_name": str(row.get("product_name", "") or ""),
                "product_category": str(row.get("product_category", "") or ""),
                "display_name": str(row.get("display_name", "") or ""),
                "display_category": str(row.get("display_category", "") or ""),
                "entity_type": str(row.get("entity_type", "") or ""),
                "aspect_labels": str(row.get("aspect_labels", "") or ""),
                "aspect_count": int(row.get("aspect_count", 0) or 0),
                "sentiment_label": str(row.get("sentiment_label", "") or ""),
                "sentiment_score": float(row.get("sentiment_score", 0.0) or 0.0),
                "review_date": str(row.get("review_date", "") or ""),
                "source_url": str(row.get("source_url", "") or ""),
            }
        )

    if not documents:
        raise RuntimeError("No sufficiently long review_text values were found for embedding.")

    embedding_backend = load_embedding_backend(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )

    log_event(
        logger,
        "embeddings_generation_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=len(documents),
        status="started",
        embedding_backend=embedding_backend.backend_name,
        embedding_model=embedding_backend.model_name,
        embedding_fallback_reason=embedding_backend.fallback_reason,
    )
    embeddings = encode_texts(
        embedding_backend,
        documents,
        batch_size=64,
        show_progress_bar=embedding_backend.backend_name == "sentence-transformers",
    )

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
            embeddings=embeddings[index:index + batch_size],
            metadatas=metadatas[index:index + batch_size],
        )

    write_embedding_backend_metadata(settings.chroma_path, embedding_backend)

    if storage_manager is not None:
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
        embedding_backend=embedding_backend.backend_name,
    )


if __name__ == "__main__":
    main()
