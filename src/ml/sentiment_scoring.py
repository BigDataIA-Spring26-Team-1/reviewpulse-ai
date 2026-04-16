"""
ReviewPulse AI sentiment scoring stage.

Run:
    poetry run python src/ml/sentiment_scoring.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings


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
    spark = build_spark()

    print("=" * 60)
    print("REVIEWPULSE AI SENTIMENT SCORING")
    print("=" * 60)

    if not settings.normalized_parquet_path.exists():
        print(f"Input parquet not found: {settings.normalized_parquet_path}")
        print("Run the Spark normalization pipeline first.")
        spark.stop()
        return

    df = spark.read.parquet(str(settings.normalized_parquet_path))

    enriched = df.withColumn("sentiment_struct", sentiment_udf(col("review_text")))
    enriched = (
        enriched
        .withColumn("sentiment_label", col("sentiment_struct.sentiment_label"))
        .withColumn("sentiment_score", col("sentiment_struct.sentiment_score"))
        .drop("sentiment_struct")
    )

    print("\nCounts by source:")
    enriched.groupBy("source").count().orderBy("source").show(truncate=False)

    print("\nSentiment by source:")
    enriched.groupBy("source", "sentiment_label").count().orderBy("source", "sentiment_label").show(truncate=False)

    print("\nAverage sentiment score by source:")
    enriched.groupBy("source").avg("sentiment_score").orderBy("source").show(truncate=False)

    if settings.sentiment_parquet_path.exists():
        shutil.rmtree(settings.sentiment_parquet_path)

    settings.sentiment_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.write.mode("overwrite").parquet(str(settings.sentiment_parquet_path))

    print(f"\nSaved sentiment-enriched parquet to: {settings.sentiment_parquet_path}")
    print("\nSample rows:")
    enriched.select(
        "source",
        "product_name",
        "sentiment_label",
        "sentiment_score",
        "review_text",
    ).show(10, truncate=100)

    spark.stop()


if __name__ == "__main__":
    main()
