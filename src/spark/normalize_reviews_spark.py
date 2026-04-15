"""
ReviewPulse AI — PySpark Normalization Pipeline
==============================================
Reads review data from multiple sources and normalizes everything
into one unified schema using Spark.

This is the Spark version of the earlier Python normalization logic.
It proves the distributed-processing architecture in the proposal.

Run:
    poetry run python src/spark/normalize_reviews_spark.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    concat_ws,
    when,
    length,
    regexp_replace,
    from_unixtime,
    to_timestamp,
    split,
    size,
)
from pyspark.sql.types import DoubleType
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "normalized_reviews_parquet")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ReviewPulse-Normalization")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def normalize_amazon(spark: SparkSession):
    amazon_path = os.path.join(DATA_DIR, "amazon_electronics_sample.jsonl")
    if not os.path.exists(amazon_path):
        return None

    df = spark.read.json(amazon_path)

    df = df.select(
        concat_ws("_", lit("amazon"), col("asin"), col("user_id")).alias("review_id"),
        when(col("parent_asin").isNotNull(), col("parent_asin"))
        .otherwise(col("asin"))
        .alias("product_name"),
        lit("electronics").alias("product_category"),
        lit("amazon").alias("source"),
        (((col("rating") - lit(1.0)) / lit(4.0)).cast(DoubleType())).alias("rating_normalized"),
        when(col("title").isNotNull(), concat_ws(". ", col("title"), col("text")))
        .otherwise(col("text"))
        .alias("review_text"),
        to_timestamp(from_unixtime((col("timestamp") / 1000).cast("bigint"))).alias("review_date"),
        col("user_id").alias("reviewer_id"),
        col("verified_purchase").alias("verified_purchase"),
        col("helpful_vote").alias("helpful_votes"),
        concat_ws("", lit("https://amazon.com/dp/"), col("asin")).alias("source_url"),
        concat_ws("", lit("Amazon Electronics Item "), col("asin")).alias("display_name"),
lit("Electronics Product").alias("display_category"),
lit("product_review").alias("entity_type"),
    )

    return df


def normalize_yelp(spark: SparkSession):
    review_path = os.path.join(DATA_DIR, "yelp", "yelp_academic_dataset_review.json")

    if not os.path.exists(review_path):
        return None

    reviews = spark.read.json(review_path)

    # Keep MVP manageable for local machine
    reviews = reviews.limit(10000)

    df = reviews.select(
        concat_ws("_", lit("yelp"), col("review_id")).alias("review_id"),
        col("business_id").alias("product_name"),
        lit("local_business").alias("product_category"),
        lit("yelp").alias("source"),
        (((col("stars") - lit(1.0)) / lit(4.0)).cast(DoubleType())).alias("rating_normalized"),
        col("text").alias("review_text"),
        to_timestamp(col("date")).alias("review_date"),
        col("user_id").alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        lit(None).cast("int").alias("helpful_votes"),
        concat_ws("", lit("https://yelp.com/biz/"), col("business_id")).alias("source_url"),
        col("business_id").alias("display_name"),
        lit("Local Business").alias("display_category"),
        lit("business_review").alias("entity_type"),
    )

    return df



def normalize_reddit(spark: SparkSession):
    reddit_path = os.path.join(DATA_DIR, "reddit_reviews.jsonl")
    if not os.path.exists(reddit_path):
        return None

    df = spark.read.json(reddit_path)

    df = df.select(
        concat_ws("_", lit("reddit"), col("source_id")).alias("review_id"),
        lit("unknown").alias("product_name"),
        when(col("subreddit").isNotNull(), col("subreddit"))
        .otherwise(lit("general"))
        .alias("product_category"),
        lit("reddit").alias("source"),
        lit(None).cast(DoubleType()).alias("rating_normalized"),
        when(col("title").isNotNull(), concat_ws(". ", col("title"), col("text")))
        .otherwise(col("text"))
        .alias("review_text"),
        to_timestamp(from_unixtime(col("created_utc").cast("bigint"))).alias("review_date"),
        col("author").alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        col("score").cast("int").alias("helpful_votes"),
        col("url").alias("source_url"),
        when(col("title").isNotNull(), col("title"))
        .otherwise(lit("Reddit Discussion"))
        .alias("display_name"),
        when(col("subreddit").isNotNull(), col("subreddit"))
        .otherwise(lit("general"))
        .alias("display_category"),
        lit("forum_post").alias("entity_type"),
    )

    return df



def normalize_youtube(spark: SparkSession):
    youtube_path = os.path.join(DATA_DIR, "youtube_reviews.jsonl")
    if not os.path.exists(youtube_path):
        return None

    df = spark.read.json(youtube_path)

    df = df.select(
        concat_ws("_", lit("youtube"), col("source_id")).alias("review_id"),
        lit("unknown").alias("product_name"),
        lit("unknown").alias("product_category"),
        lit("youtube").alias("source"),
        lit(None).cast(DoubleType()).alias("rating_normalized"),
        col("text").alias("review_text"),
        to_timestamp(from_unixtime(col("created_utc").cast("bigint"))).alias("review_date"),
        col("channel").alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        lit(None).cast("int").alias("helpful_votes"),
        col("url").alias("source_url"),
        lit("YouTube Review").alias("display_name"),
        lit("Video Review").alias("display_category"),
        lit("video_transcript").alias("entity_type"),
    )

    return df


def add_text_length(df):
    return df.withColumn(
        "text_length_words",
        size(split(regexp_replace(col("review_text"), r"\s+", " "), " "))
    )


def main():
    spark = build_spark()
    print("=" * 60)
    print("REVIEWPULSE AI — PYSPARK NORMALIZATION PIPELINE")
    print("=" * 60)

    dataframes = []

    amazon_df = normalize_amazon(spark)
    if amazon_df is not None:
        print("Loaded Amazon sample data")
        dataframes.append(amazon_df)

    yelp_df = normalize_yelp(spark)
    if yelp_df is not None:
        print("Loaded Yelp review data")
        dataframes.append(yelp_df)

    reddit_df = normalize_reddit(spark)
    if reddit_df is not None:
        print("Loaded Reddit data")
        dataframes.append(reddit_df)

    youtube_df = normalize_youtube(spark)
    if youtube_df is not None:
        print("Loaded YouTube transcript data")
        dataframes.append(youtube_df)

    if not dataframes:
        print("No input data found.")
        spark.stop()
        return

    unified_df = dataframes[0]
    for df in dataframes[1:]:
        unified_df = unified_df.unionByName(df)

    unified_df = add_text_length(unified_df)

    print("\nSchema:")
    unified_df.printSchema()

    print("\nRecord counts by source:")
    unified_df.groupBy("source").count().show(truncate=False)

    print("\nNull counts for key fields:")
    for field in ["rating_normalized", "verified_purchase", "helpful_votes", "review_date"]:
        null_count = unified_df.filter(col(field).isNull()).count()
        print(f"  {field}: {null_count}")

    print("\nAverage text length by source:")
    unified_df.groupBy("source").avg("text_length_words").show(truncate=False)

    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    unified_df.write.mode("overwrite").parquet(OUTPUT_DIR)

    print(f"\nSaved parquet output to: {OUTPUT_DIR}")
    print("\nSample rows:")
    unified_df.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()  