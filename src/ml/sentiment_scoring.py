"""
ReviewPulse AI sentiment scoring stage.

Run:
    poetry run python src/ml/sentiment_scoring.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

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


def build_spark() -> SparkSession:
    settings = get_settings()
    return (
        SparkSession.builder
        .appName("ReviewPulse-Sentiment")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
        .getOrCreate()
    )


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
    spark = build_spark()

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
        spark.stop()
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

    df = spark.read.parquet(str(settings.normalized_parquet_path))

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

    spark.stop()
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
