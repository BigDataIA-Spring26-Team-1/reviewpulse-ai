"""
ReviewPulse AI sentiment scoring stage.

Run:
    poetry run python src/ml/sentiment_scoring.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections.abc import Callable
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
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path / "part-00000.parquet", compression="snappy")


def _sentiment_part_index(path: Path) -> int | None:
    if not path.name.startswith("part-") or path.suffix != ".parquet":
        return None
    try:
        return int(path.stem.removeprefix("part-"))
    except ValueError:
        return None


def _find_resumable_sentiment_parts(
    output_path: Path,
    *,
    expected_batch_size: int,
    expected_total_rows: int | None = None,
) -> tuple[int, int]:
    if not output_path.exists():
        return 0, 0

    indexed_parts = sorted(
        (index, path)
        for path in output_path.glob("part-*.parquet")
        if (index := _sentiment_part_index(path)) is not None
    )
    if not indexed_parts:
        return 0, 0

    expected_indices = list(range(len(indexed_parts)))
    actual_indices = [index for index, _path in indexed_parts]
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Cannot resume sentiment scoring because output parquet parts are not contiguous "
            f"from part-00000: {actual_indices[:10]}"
        )

    existing_rows = 0
    for index, part_path in indexed_parts:
        row_count = pq.ParquetFile(part_path).metadata.num_rows
        if row_count <= 0:
            raise RuntimeError(f"Cannot resume sentiment scoring from empty parquet part: {part_path}")
        final_complete_part = (
            expected_total_rows is not None
            and existing_rows + row_count == expected_total_rows
            and row_count <= expected_batch_size
        )
        if row_count != expected_batch_size and not final_complete_part:
            raise RuntimeError(
                "Cannot safely resume sentiment scoring because the existing final part is "
                f"not a full batch: {part_path} has {row_count} rows."
            )
        existing_rows += row_count

    return len(indexed_parts), existing_rows


def _env_flag_enabled(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    return raw_value not in {"0", "false", "no", "off"}


def _aspect_backend_mode() -> str:
    return os.getenv("ASPECT_EXTRACTION_BACKEND", "heuristic").strip().lower() or "heuristic"


def _should_probe_ollama_aspects() -> bool:
    return _env_flag_enabled("ASPECT_EXTRACTION_ENABLED", True) and _aspect_backend_mode() in {
        "auto",
        "ollama",
    }


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
    row_batch_size: int | None = None,
    resume: bool | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    progress_interval_parts: int | None = None,
) -> int:
    dataset = ds.dataset(str(input_path), format="parquet")
    resolved_batch_size = int(row_batch_size or os.getenv("SENTIMENT_ROW_BATCH_SIZE", "50000"))
    resolved_batch_size = max(1, resolved_batch_size)
    aspect_enabled = _env_flag_enabled("ASPECT_EXTRACTION_ENABLED", True)
    aspect_backend = _aspect_backend_mode()

    backend = sentiment_backend or load_sentiment_backend(
        sentiment_model_name or DEFAULT_SENTIMENT_MODEL
    )
    resolved_ollama_host = str(ollama_host or "http://localhost:11434")
    resolved_ollama_model = str(ollama_model or "llama3.1:8b")
    resolved_ollama_timeout = int(ollama_timeout_seconds or 30)
    should_use_ollama = (
        aspect_backend in {"auto", "ollama"}
        and probe_ollama_host(resolved_ollama_host, resolved_ollama_timeout)
        if ollama_available is None
        else aspect_backend in {"auto", "ollama"} and ollama_available
    )
    resume_enabled = _env_flag_enabled("SENTIMENT_RESUME", False) if resume is None else resume
    progress_interval = max(
        1,
        int(progress_interval_parts or os.getenv("SENTIMENT_PROGRESS_INTERVAL_PARTS", "10")),
    )

    skipped_batches = 0
    existing_rows = 0
    if resume_enabled:
        skipped_batches, existing_rows = _find_resumable_sentiment_parts(
            output_path,
            expected_batch_size=resolved_batch_size,
            expected_total_rows=dataset.count_rows(),
        )
    elif output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    total_rows = existing_rows
    part_index = skipped_batches
    if progress_callback and skipped_batches:
        progress_callback(
            {
                "part_count": part_index,
                "record_count": total_rows,
                "skipped_batches": skipped_batches,
            }
        )

    for batch_index, record_batch in enumerate(dataset.to_batches(batch_size=resolved_batch_size)):
        if batch_index < skipped_batches:
            continue

        rows = pa.Table.from_batches([record_batch]).to_pylist()
        if not rows:
            continue

        sentiment_scores = score_texts(
            backend,
            [str(row.get("review_text", "") or "") for row in rows],
        )

        enriched_rows: list[dict[str, object]] = []
        for row, (sentiment_label, sentiment_score) in zip(rows, sentiment_scores):
            review_text = str(row.get("review_text", "") or "")
            aspects = (
                _extract_aspects_for_text(
                    review_text,
                    ollama_available=should_use_ollama,
                    ollama_host=resolved_ollama_host,
                    ollama_model=resolved_ollama_model,
                    ollama_timeout_seconds=resolved_ollama_timeout,
                )
                if aspect_enabled
                else []
            )
            aspect_labels, aspect_count, aspect_details_json = serialize_aspects(aspects)

            enriched_row = dict(row)
            enriched_row["aspect_labels"] = aspect_labels
            enriched_row["aspect_count"] = int(aspect_count)
            enriched_row["aspect_details_json"] = aspect_details_json
            enriched_row["sentiment_label"] = sentiment_label
            enriched_row["sentiment_score"] = float(sentiment_score)
            enriched_rows.append(enriched_row)

        table = pa.Table.from_pylist(enriched_rows)
        pq.write_table(
            table,
            output_path / f"part-{part_index:05d}.parquet",
            compression="snappy",
        )
        total_rows += len(enriched_rows)
        part_index += 1
        if progress_callback and (
            part_index == skipped_batches + 1
            or part_index % progress_interval == 0
        ):
            progress_callback(
                {
                    "part_count": part_index,
                    "record_count": total_rows,
                    "skipped_batches": skipped_batches,
                }
            )

    if total_rows == 0:
        _write_parquet_rows_with_arrow([], output_path)
    return total_rows


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
    ollama_available = (
        probe_ollama_host(
            settings.ollama_host,
            settings.ollama_timeout_seconds,
        )
        if _should_probe_ollama_aspects()
        else False
    )
    aspect_backend = (
        "disabled"
        if not _env_flag_enabled("ASPECT_EXTRACTION_ENABLED", True)
        else "ollama" if ollama_available else "heuristic"
    )
    resume_enabled = _env_flag_enabled("SENTIMENT_RESUME", False)

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
        aspect_backend=aspect_backend,
        resume_enabled=resume_enabled,
    )

    def log_sentiment_progress(progress: dict[str, int]) -> None:
        log_event(
            logger,
            "sentiment_scoring_progress",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="running",
            part_count=progress["part_count"],
            record_count=progress["record_count"],
            skipped_batches=progress.get("skipped_batches", 0),
        )

    enriched_count = _score_sentiment_with_arrow(
        settings.normalized_parquet_path,
        settings.sentiment_parquet_path,
        sentiment_backend=sentiment_backend,
        ollama_available=ollama_available,
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
        resume=resume_enabled,
        progress_callback=log_sentiment_progress,
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
        aspect_backend=aspect_backend,
    )


if __name__ == "__main__":
    main()
