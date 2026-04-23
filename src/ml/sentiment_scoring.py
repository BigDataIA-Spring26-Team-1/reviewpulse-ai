"""
ReviewPulse AI sentiment scoring stage.

Run:
    poetry run python src/ml/sentiment_scoring.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_context import build_run_context
from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ml.aspect_extraction import (
    extract_aspects_heuristic,
    extract_aspects_with_ollama,
    probe_ollama_host,
    serialize_aspects,
)
from src.ml.sentiment_backend import DEFAULT_SENTIMENT_MODEL, SentimentBackend, load_sentiment_backend, score_texts


def _read_parquet_rows_with_arrow(input_path: Path) -> list[dict[str, object]]:
    dataset = ds.dataset(str(input_path), format="parquet")
    table = dataset.to_table()
    return [dict(row) for row in table.to_pylist()]


def _write_parquet_rows_with_arrow(rows: list[dict[str, object]], output_path: Path) -> None:
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path / "part-00000.parquet", compression="snappy")


def _extract_aspects_for_text(
    text: str | None,
    *,
    ollama_available: bool,
    ollama_host: str,
    ollama_model: str,
    ollama_timeout_seconds: int,
) -> list[dict[str, object]]:
    if ollama_available:
        ollama_aspects = extract_aspects_with_ollama(
            text,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            timeout_seconds=ollama_timeout_seconds,
        )
        if ollama_aspects is not None:
            return ollama_aspects
    return extract_aspects_heuristic(text)


def _score_sentiment_with_arrow(
    input_path: Path,
    output_path: Path,
    *,
    sentiment_backend: SentimentBackend | None = None,
    sentiment_model_name: str | None = None,
    ollama_available: bool | None = None,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    ollama_timeout_seconds: int | None = None,
) -> int:
    rows = _read_parquet_rows_with_arrow(input_path)
    if not rows:
        _write_parquet_rows_with_arrow([], output_path)
        return 0

    backend = sentiment_backend or load_sentiment_backend(
        sentiment_model_name or DEFAULT_SENTIMENT_MODEL
    )
    resolved_ollama_host = str(ollama_host or "http://localhost:11434")
    resolved_ollama_model = str(ollama_model or "llama3.1:8b")
    resolved_ollama_timeout = int(ollama_timeout_seconds or 30)
    should_use_ollama = (
        probe_ollama_host(resolved_ollama_host, resolved_ollama_timeout)
        if ollama_available is None
        else ollama_available
    )

    sentiment_scores = score_texts(
        backend,
        [str(row.get("review_text", "") or "") for row in rows],
    )

    enriched_rows: list[dict[str, object]] = []
    for row, (sentiment_label, sentiment_score) in zip(rows, sentiment_scores):
        review_text = str(row.get("review_text", "") or "")
        aspects = _extract_aspects_for_text(
            review_text,
            ollama_available=should_use_ollama,
            ollama_host=resolved_ollama_host,
            ollama_model=resolved_ollama_model,
            ollama_timeout_seconds=resolved_ollama_timeout,
        )
        aspect_labels, aspect_count, aspect_details_json = serialize_aspects(aspects)

        enriched_row = dict(row)
        enriched_row["aspect_labels"] = aspect_labels
        enriched_row["aspect_count"] = int(aspect_count)
        enriched_row["aspect_details_json"] = aspect_details_json
        enriched_row["sentiment_label"] = sentiment_label
        enriched_row["sentiment_score"] = float(sentiment_score)
        enriched_rows.append(enriched_row)

    _write_parquet_rows_with_arrow(enriched_rows, output_path)
    return len(enriched_rows)


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("ml.sentiment")
    run_context = build_run_context(stage="sentiment_scoring")
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

    if not settings.normalized_parquet_path.exists():
        log_event(
            logger,
            "sentiment_scoring_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(settings.normalized_parquet_path),
            status="failed",
            error_type="MissingInputError",
            error_message="Normalized parquet not found. Run the Spark normalization pipeline first.",
        )
        raise RuntimeError("Normalized parquet not found. Run the Spark normalization pipeline first.")

    sentiment_backend = load_sentiment_backend(settings.sentiment_model)
    ollama_available = probe_ollama_host(
        settings.ollama_host,
        settings.ollama_timeout_seconds,
    )

    log_event(
        logger,
        "sentiment_scoring_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(settings.normalized_parquet_path),
        status="started",
        sentiment_backend=sentiment_backend.backend_name,
        sentiment_model=sentiment_backend.model_name,
        sentiment_fallback_reason=sentiment_backend.fallback_reason,
        aspect_backend="ollama" if ollama_available else "heuristic",
    )

    enriched_count = _score_sentiment_with_arrow(
        settings.normalized_parquet_path,
        settings.sentiment_parquet_path,
        sentiment_backend=sentiment_backend,
        ollama_available=ollama_available,
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
    )

    if storage_manager is not None:
        run_prefix = storage_manager.resolver.processed_run_prefix("sentiment_parquet", run_context.run_id)
        current_prefix = storage_manager.resolver.processed_current_prefix("sentiment_parquet")

        log_event(
            logger,
            "s3_upload_started",
            stage="sentiment_parquet",
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(settings.sentiment_parquet_path),
            output_path=run_prefix,
            status="started",
        )
        uploaded = storage_manager.upload_directory(settings.sentiment_parquet_path, run_prefix)
        log_event(
            logger,
            "s3_upload_completed",
            stage="sentiment_parquet",
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            input_path=str(settings.sentiment_parquet_path),
            output_path=run_prefix,
            file_count=len(uploaded),
            status="success",
        )
        promotion = storage_manager.promote_run_prefix(
            run_prefix,
            current_prefix,
            run_id=run_context.run_id,
            metadata={"stage": "sentiment_parquet", "status": "success"},
        )
        log_event(
            logger,
            "latest_run_promoted",
            stage="sentiment_parquet",
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            output_path=current_prefix,
            file_count=promotion["copied_count"],
            status="success",
        )

    log_event(
        logger,
        "sentiment_scoring_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=enriched_count,
        output_path=str(settings.sentiment_parquet_path),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
        sentiment_backend=sentiment_backend.backend_name,
        aspect_backend="ollama" if ollama_available else "heuristic",
    )


if __name__ == "__main__":
    main()
