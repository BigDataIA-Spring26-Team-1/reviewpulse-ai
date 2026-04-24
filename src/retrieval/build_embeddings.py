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
from collections.abc import Iterator
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


def _resolve_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _normalize_embedding_row(raw_row: dict[str, object]) -> dict[str, object]:
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
    return row


def _iter_embedding_batches_with_arrow(
    input_path: Path,
    *,
    row_batch_size: int | None = None,
) -> Iterator[list[dict[str, object]]]:
    dataset = ds.dataset(str(input_path), format="parquet")
    available_columns = set(dataset.schema.names)
    requested_columns = [column for column in EMBEDDING_COLUMNS if column in available_columns]
    resolved_batch_size = row_batch_size or _resolve_positive_int_env("EMBEDDING_ROW_BATCH_SIZE", 5000)

    for record_batch in dataset.to_batches(
        columns=requested_columns,
        batch_size=resolved_batch_size,
    ):
        rows = [
            _normalize_embedding_row(dict(raw_row))
            for raw_row in record_batch.to_pylist()
        ]
        if rows:
            yield rows


def _load_embedding_rows_with_arrow(input_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch_rows in _iter_embedding_batches_with_arrow(input_path):
        rows.extend(batch_rows)
    return rows


def _build_chroma_payload(row: dict[str, object]) -> tuple[str, str, dict[str, object]] | None:
    review_id = str(row.get("review_id", "") or "").strip()
    review_text = str(row.get("review_text", "") or "")
    if not review_id or len(review_text.strip()) < 20:
        return None

    metadata = {
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
    return review_id, review_text, metadata


def _upsert_embedding_payloads(
    collection: Any,
    embedding_backend: Any,
    payloads: list[tuple[str, str, dict[str, object]]],
    *,
    encode_batch_size: int,
    chroma_batch_size: int,
) -> int:
    if not payloads:
        return 0

    ids = [payload[0] for payload in payloads]
    documents = [payload[1] for payload in payloads]
    metadatas = [payload[2] for payload in payloads]
    embeddings = encode_texts(
        embedding_backend,
        documents,
        batch_size=encode_batch_size,
        show_progress_bar=False,
    )

    for index in range(0, len(documents), chroma_batch_size):
        collection.upsert(
            ids=ids[index:index + chroma_batch_size],
            documents=documents[index:index + chroma_batch_size],
            embeddings=embeddings[index:index + chroma_batch_size],
            metadatas=metadatas[index:index + chroma_batch_size],
        )
    return len(documents)


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

    embedding_backend = load_embedding_backend(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    row_batch_size = _resolve_positive_int_env("EMBEDDING_ROW_BATCH_SIZE", 5000)
    encode_batch_size = _resolve_positive_int_env("EMBEDDING_ENCODE_BATCH_SIZE", 64)
    chroma_batch_size = _resolve_positive_int_env("CHROMA_ADD_BATCH_SIZE", 500)

    log_event(
        logger,
        "embeddings_generation_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(settings.sentiment_parquet_path),
        row_batch_size=row_batch_size,
        encode_batch_size=encode_batch_size,
        chroma_batch_size=chroma_batch_size,
        status="started",
        embedding_backend=embedding_backend.backend_name,
        embedding_model=embedding_backend.model_name,
        embedding_fallback_reason=embedding_backend.fallback_reason,
    )

    if settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)

    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION_NAME", "reviewpulse_reviews")
    )

    total_input_rows = 0
    total_embedded_rows = 0
    last_logged_rows = 0
    progress_interval = _resolve_positive_int_env("EMBEDDING_PROGRESS_INTERVAL_ROWS", 50000)
    for rows in _iter_embedding_batches_with_arrow(
        settings.sentiment_parquet_path,
        row_batch_size=row_batch_size,
    ):
        total_input_rows += len(rows)
        payloads = [
            payload
            for row in rows
            if (payload := _build_chroma_payload(row)) is not None
        ]
        total_embedded_rows += _upsert_embedding_payloads(
            collection,
            embedding_backend,
            payloads,
            encode_batch_size=encode_batch_size,
            chroma_batch_size=chroma_batch_size,
        )

        if total_embedded_rows - last_logged_rows >= progress_interval:
            last_logged_rows = total_embedded_rows
            log_event(
                logger,
                "embeddings_generation_progress",
                stage=run_context.stage,
                run_id=run_context.run_id,
                dag_id=run_context.dag_id,
                task_id=run_context.task_id,
                input_rows_seen=total_input_rows,
                embedded_rows=total_embedded_rows,
                status="running",
            )

    if total_input_rows == 0:
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
    if total_embedded_rows == 0:
        raise RuntimeError("No sufficiently long review_text values were found for embedding.")

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
        record_count=total_embedded_rows,
        input_record_count=total_input_rows,
        output_path=str(settings.chroma_path),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
        embedding_backend=embedding_backend.backend_name,
    )


if __name__ == "__main__":
    main()
