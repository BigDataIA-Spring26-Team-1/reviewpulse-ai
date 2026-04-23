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
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.common.run_context import build_run_context
from src.common.spark_runtime import ensure_local_hadoop_home


POSITIVE_WORDS = {
    "great",
    "excellent",
    "amazing",
    "good",
    "love",
    "best",
    "awesome",
    "perfect",
    "premium",
    "smooth",
    "fast",
    "recommend",
    "durable",
    "comfortable",
    "improved",
    "incredible",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "poor",
    "worst",
    "hate",
    "broken",
    "cheap",
    "slow",
    "disappointing",
    "lag",
    "issue",
    "problem",
    "expensive",
    "overpriced",
    "refund",
    "return",
}

WINDOWS_NATIVE_PARQUET_ERROR_MARKERS = (
    "NativeIO$Windows.access0",
    "UnsatisfiedLinkError",
    "Access denied",
)


def build_spark() -> SparkSession:
    settings = get_settings()
    hadoop_home = ensure_local_hadoop_home(PROJECT_ROOT)
    builder = (
        SparkSession.builder
        .appName("ReviewPulse-Sentiment")
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


def _score_sentiment_with_arrow(input_path: Path, output_path: Path) -> int:
    dataset = ds.dataset(str(input_path), format="parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    total_records = 0
    wrote_any_batches = False
    for part_index, batch in enumerate(dataset.to_batches()):
        wrote_any_batches = True
        table = pa.Table.from_batches([batch])
        sentiments = [score_sentiment(text) for text in table.column("review_text").to_pylist()]
        labels = pa.array([label for label, _score in sentiments], type=pa.string())
        scores = pa.array([score for _label, score in sentiments], type=pa.float64())
        table = table.append_column("sentiment_label", labels)
        table = table.append_column("sentiment_score", scores)
        pq.write_table(table, output_path / f"part-{part_index:05d}.parquet", compression="snappy")
        total_records += table.num_rows

    if not wrote_any_batches:
        empty_table = pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in dataset.schema],
            names=dataset.schema.names,
        )
        empty_table = empty_table.append_column("sentiment_label", pa.array([], type=pa.string()))
        empty_table = empty_table.append_column("sentiment_score", pa.array([], type=pa.float64()))
        pq.write_table(empty_table, output_path / "part-00000.parquet", compression="snappy")

    return total_records


def score_sentiment(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return ("neutral", 0.0)

    words = text.lower().split()
    pos = sum(1 for word in words if word.strip(".,!?;:()[]'\"") in POSITIVE_WORDS)
    neg = sum(1 for word in words if word.strip(".,!?;:()[]'\"") in NEGATIVE_WORDS)

    total = pos + neg
    if total == 0:
        return ("neutral", 0.0)

    score = (pos - neg) / total
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return (label, float(round(score, 4)))


sentiment_schema = StructType(
    [
        StructField("sentiment_label", StringType(), True),
        StructField("sentiment_score", DoubleType(), True),
    ]
)

sentiment_udf = udf(score_sentiment, sentiment_schema)


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("ml.sentiment")
    run_context = build_run_context(stage="sentiment_scoring")
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

    log_event(
        logger,
        "sentiment_scoring_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(settings.normalized_parquet_path),
        status="started",
    )
    spark: SparkSession | None = None
    try:
        spark = build_spark()
        df = spark.read.parquet(str(settings.normalized_parquet_path))
        enriched = df.withColumn("sentiment_struct", sentiment_udf(col("review_text")))
        enriched = (
            enriched
            .withColumn("sentiment_label", col("sentiment_struct.sentiment_label"))
            .withColumn("sentiment_score", col("sentiment_struct.sentiment_score"))
            .drop("sentiment_struct")
        )
        enriched_count = enriched.count()

        if settings.sentiment_parquet_path.exists():
            shutil.rmtree(settings.sentiment_parquet_path)

        settings.sentiment_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.write.mode("overwrite").parquet(str(settings.sentiment_parquet_path))
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
            input_path=str(settings.normalized_parquet_path),
            output_path=str(settings.sentiment_parquet_path),
            status="fallback",
            error_type=type(exc).__name__,
            error_message="Spark parquet read failed on Windows; falling back to pyarrow.",
        )
        enriched_count = _score_sentiment_with_arrow(
            settings.normalized_parquet_path,
            settings.sentiment_parquet_path,
        )
    finally:
        if spark is not None:
            spark.stop()
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
    )


if __name__ == "__main__":
    main()
