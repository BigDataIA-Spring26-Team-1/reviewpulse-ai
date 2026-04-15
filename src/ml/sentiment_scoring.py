"""
ReviewPulse AI — MVP Sentiment Scoring
=====================================
Reads normalized reviews and adds sentiment_label + sentiment_score.

Run:
    poetry run python src/ml/sentiment_scoring.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "normalized_reviews_parquet")
OUTPUT_PATH = os.path.join(DATA_DIR, "reviews_with_sentiment_parquet")


POSITIVE_WORDS = {
    "great", "excellent", "amazing", "good", "love", "best", "awesome",
    "perfect", "premium", "smooth", "fast", "recommend", "durable",
    "comfortable", "improved", "incredible"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "poor", "worst", "hate", "broken",
    "cheap", "slow", "disappointing", "lag", "issue", "problem",
    "expensive", "overpriced", "refund", "return"
}


def build_spark():
    return (
        SparkSession.builder
        .appName("ReviewPulse-Sentiment")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def score_sentiment(text: str):
    if not text or not text.strip():
        return ("neutral", 0.0)

    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?;:()[]'\"") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?;:()[]'\"") in NEGATIVE_WORDS)

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


sentiment_schema = StructType([
    StructField("sentiment_label", StringType(), True),
    StructField("sentiment_score", DoubleType(), True),
])

sentiment_udf = udf(score_sentiment, sentiment_schema)


def main():
    spark = build_spark()

    print("=" * 60)
    print("REVIEWPULSE AI — SENTIMENT SCORING")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        print(f"Input parquet not found: {INPUT_PATH}")
        print("Run the PySpark normalization pipeline first.")
        spark.stop()
        return

    df = spark.read.parquet(INPUT_PATH)

    enriched = df.withColumn("sentiment_struct", sentiment_udf(col("review_text")))
    enriched = (
        enriched
        .withColumn("sentiment_label", col("sentiment_struct.sentiment_label"))
        .withColumn("sentiment_score", col("sentiment_struct.sentiment_score"))
        .drop("sentiment_struct")
    )

    print("\nCounts by source:")
    enriched.groupBy("source").count().show(truncate=False)

    print("\nSentiment by source:")
    enriched.groupBy("source", "sentiment_label").count().orderBy("source", "sentiment_label").show(truncate=False)

    print("\nAverage sentiment score by source:")
    enriched.groupBy("source").avg("sentiment_score").show(truncate=False)

    if os.path.exists(OUTPUT_PATH):
        import shutil
        shutil.rmtree(OUTPUT_PATH)

    enriched.write.mode("overwrite").parquet(OUTPUT_PATH)

    print(f"\nSaved sentiment-enriched parquet to: {OUTPUT_PATH}")
    print("\nSample rows:")
    enriched.select(
        "source", "product_name", "sentiment_label", "sentiment_score", "review_text"
    ).show(10, truncate=100)

    spark.stop()


if __name__ == "__main__":
    main()