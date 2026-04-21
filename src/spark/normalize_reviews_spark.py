"""
ReviewPulse AI Spark normalization pipeline.

This is the normalization stage for the core sources:
Amazon, Yelp, eBay, iFixit, and YouTube. Reddit remains optional.

Run:
    poetry run python src/spark/normalize_reviews_spark.py
"""

from __future__ import annotations

import importlib
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEPENDENCY_SETUP_HINT = (
    "Install project dependencies with `poetry install` and use the Poetry environment "
    "before running the Spark normalization pipeline."
)

try:
    pyspark_sql = importlib.import_module("pyspark.sql")
    pyspark_functions = importlib.import_module("pyspark.sql.functions")
    pyspark_types = importlib.import_module("pyspark.sql.types")

    DataFrame = pyspark_sql.DataFrame
    SparkSession = pyspark_sql.SparkSession
    coalesce = pyspark_functions.coalesce
    col = pyspark_functions.col
    concat_ws = pyspark_functions.concat_ws
    from_unixtime = pyspark_functions.from_unixtime
    greatest = pyspark_functions.greatest
    least = pyspark_functions.least
    length = pyspark_functions.length
    lit = pyspark_functions.lit
    regexp_replace = pyspark_functions.regexp_replace
    size = pyspark_functions.size
    split = pyspark_functions.split
    to_timestamp = pyspark_functions.to_timestamp
    trim = pyspark_functions.trim
    when = pyspark_functions.when
    DoubleType = pyspark_types.DoubleType
    PYSPARK_AVAILABLE = True
except ImportError:
    DataFrame = Any
    SparkSession = Any
    PYSPARK_AVAILABLE = False

    def _missing_pyspark(*_args, **_kwargs):
        raise RuntimeError("Missing dependency: `pyspark`. " + DEPENDENCY_SETUP_HINT)

    coalesce = _missing_pyspark
    col = _missing_pyspark
    concat_ws = _missing_pyspark
    from_unixtime = _missing_pyspark
    greatest = _missing_pyspark
    least = _missing_pyspark
    length = _missing_pyspark
    lit = _missing_pyspark
    regexp_replace = _missing_pyspark
    size = _missing_pyspark
    split = _missing_pyspark
    to_timestamp = _missing_pyspark
    trim = _missing_pyspark
    when = _missing_pyspark
    DoubleType = _missing_pyspark


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.common.run_context import PipelineRunContext, build_run_context
from src.normalization.core import (
    AMAZON_FILE_CANDIDATES,
    EBAY_FILE_CANDIDATES,
    IFIXIT_FILE_CANDIDATES,
    REDDIT_FILE_CANDIDATES,
    YELP_BUSINESS_FILE_CANDIDATES,
    YELP_REVIEW_FILE_CANDIDATES,
    YOUTUBE_FILE_CANDIDATES,
    find_first_existing_path,
    resolve_source_input_path,
)


def build_spark() -> Any:
    if not PYSPARK_AVAILABLE:
        raise RuntimeError("Missing dependency: `pyspark`. " + DEPENDENCY_SETUP_HINT)
    settings = get_settings()
    return (
        SparkSession.builder
        .appName("ReviewPulse-Normalization")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
        .getOrCreate()
    )


def _read_json(spark: Any, candidates: tuple[str, ...]) -> tuple[Any | None, Path | None]:
    settings = get_settings()
    source_name = "amazon" if candidates == AMAZON_FILE_CANDIDATES else ""
    path = resolve_source_input_path(settings.data_dir, source_name, candidates)
    if not path:
        return None, None
    if path.is_dir():
        jsonl_paths = sorted(
            str(candidate.resolve())
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() == ".jsonl"
        )
        if not jsonl_paths:
            return None, path
        return spark.read.json(jsonl_paths), path
    return spark.read.json(str(path)), path


def _scale_column(column_name: str, minimum: float, maximum: float):
    return when(
        col(column_name).isNull(),
        lit(None).cast(DoubleType()),
    ).otherwise(
        greatest(
            lit(0.0),
            least(
                lit(1.0),
                (col(column_name).cast(DoubleType()) - lit(minimum)) / lit(maximum - minimum),
            ),
        )
    )


def normalize_amazon(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, AMAZON_FILE_CANDIDATES)
    if df is None:
        return None, None

    amazon_df = df.select(
        concat_ws("_", lit("amazon"), col("asin"), col("user_id")).alias("review_id"),
        coalesce(col("parent_asin"), col("asin")).alias("product_name"),
        lit("electronics").alias("product_category"),
        lit("amazon").alias("source"),
        _scale_column("rating", 1.0, 5.0).alias("rating_normalized"),
        when(
            col("title").isNotNull() & (trim(col("title")) != ""),
            concat_ws(". ", col("title"), coalesce(col("text"), lit(""))),
        ).otherwise(coalesce(col("text"), lit(""))).alias("review_text"),
        to_timestamp(from_unixtime((col("timestamp") / lit(1000)).cast("bigint"))).alias("review_date"),
        coalesce(col("user_id"), lit("unknown")).alias("reviewer_id"),
        col("verified_purchase").alias("verified_purchase"),
        col("helpful_vote").cast("int").alias("helpful_votes"),
        when(
            col("url").isNotNull() & (trim(col("url")) != ""),
            col("url"),
        ).otherwise(concat_ws("", lit("https://amazon.com/dp/"), col("asin"))).alias("source_url"),
        when(
            col("title").isNotNull() & (trim(col("title")) != ""),
            col("title"),
        ).otherwise(concat_ws("", lit("Amazon Electronics Item "), col("asin"))).alias("display_name"),
        lit("Electronics Product").alias("display_category"),
        lit("product_review").alias("entity_type"),
    )
    return amazon_df, path


def normalize_yelp(spark: Any) -> tuple[Any | None, Path | None]:
    reviews, review_path = _read_json(spark, YELP_REVIEW_FILE_CANDIDATES)
    if reviews is None:
        return None, None

    business_path = find_first_existing_path(get_settings().data_dir, YELP_BUSINESS_FILE_CANDIDATES)
    if business_path:
        businesses = spark.read.json(str(business_path)).select("business_id", "name", "categories")
        reviews = reviews.join(businesses, on="business_id", how="left")

    categories = when(
        col("categories").isNotNull() & (trim(col("categories")) != ""),
        col("categories"),
    ).otherwise(lit("local_business"))

    yelp_df = reviews.select(
        concat_ws("_", lit("yelp"), col("review_id")).alias("review_id"),
        coalesce(col("name"), col("business_id")).alias("product_name"),
        categories.alias("product_category"),
        lit("yelp").alias("source"),
        _scale_column("stars", 1.0, 5.0).alias("rating_normalized"),
        coalesce(col("text"), lit("")).alias("review_text"),
        to_timestamp(col("date")).alias("review_date"),
        coalesce(col("user_id"), lit("unknown")).alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        lit(None).cast("int").alias("helpful_votes"),
        concat_ws("", lit("https://yelp.com/biz/"), col("business_id")).alias("source_url"),
        coalesce(col("name"), col("business_id")).alias("display_name"),
        when(
            col("categories").isNotNull() & (trim(col("categories")) != ""),
            split(col("categories"), ",").getItem(0),
        ).otherwise(lit("Local Business")).alias("display_category"),
        lit("business_review").alias("entity_type"),
    )
    return yelp_df, review_path


def normalize_ebay(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, EBAY_FILE_CANDIDATES)
    if df is None:
        return None, None

    ebay_df = df.select(
        concat_ws("_", lit("ebay"), coalesce(col("item_id"), lit("unknown"))).alias("review_id"),
        coalesce(col("title"), col("item_title"), col("item_id")).alias("product_name"),
        coalesce(col("category"), col("primary_category"), lit("unknown")).alias("product_category"),
        lit("ebay").alias("source"),
        _scale_column("seller_rating", 0.0, 100.0).alias("rating_normalized"),
        coalesce(col("feedback_text"), col("review_text"), lit("")).alias("review_text"),
        to_timestamp(coalesce(col("listing_date"), col("review_date"))).alias("review_date"),
        coalesce(col("seller_id"), col("seller"), col("reviewer_id"), lit("unknown")).alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        coalesce(col("feedback_count"), col("helpful_votes")).cast("int").alias("helpful_votes"),
        when(
            col("url").isNotNull() & (trim(col("url")) != ""),
            col("url"),
        ).otherwise(concat_ws("", lit("https://www.ebay.com/itm/"), col("item_id"))).alias("source_url"),
        coalesce(col("title"), col("item_title"), col("item_id")).alias("display_name"),
        coalesce(col("category"), col("primary_category"), lit("Marketplace Listing")).alias("display_category"),
        lit("listing_review").alias("entity_type"),
    )
    return ebay_df, path


def normalize_ifixit(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, IFIXIT_FILE_CANDIDATES)
    if df is None:
        return None, None

    ifixit_df = df.select(
        concat_ws("_", lit("ifixit"), coalesce(col("guide_id"), col("source_id"), lit("unknown"))).alias("review_id"),
        coalesce(col("device_name"), col("title"), col("guide_id")).alias("product_name"),
        coalesce(col("device_category"), lit("repairability")).alias("product_category"),
        lit("ifixit").alias("source"),
        _scale_column("repairability_score", 1.0, 10.0).alias("rating_normalized"),
        coalesce(col("review_text"), col("text"), lit("")).alias("review_text"),
        to_timestamp(col("published_date")).alias("review_date"),
        coalesce(col("author"), lit("unknown")).alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        col("helpful_votes").cast("int").alias("helpful_votes"),
        when(
            col("url").isNotNull() & (trim(col("url")) != ""),
            col("url"),
        ).otherwise(concat_ws("", lit("https://www.ifixit.com/Guide/"), coalesce(col("guide_id"), lit("unknown")))).alias("source_url"),
        coalesce(col("title"), col("device_name"), col("guide_id")).alias("display_name"),
        coalesce(col("device_category"), lit("Repair Guide")).alias("display_category"),
        lit("repair_review").alias("entity_type"),
    )
    return ifixit_df, path


def normalize_youtube(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, YOUTUBE_FILE_CANDIDATES)
    if df is None:
        return None, None

    youtube_df = df.select(
        concat_ws("_", lit("youtube"), coalesce(col("source_id"), col("video_id"), lit("unknown"))).alias("review_id"),
        coalesce(col("product_name"), lit("unknown")).alias("product_name"),
        coalesce(col("product_category"), lit("unknown")).alias("product_category"),
        lit("youtube").alias("source"),
        lit(None).cast(DoubleType()).alias("rating_normalized"),
        coalesce(col("text"), col("transcript"), lit("")).alias("review_text"),
        coalesce(
            to_timestamp(col("published_date")),
            to_timestamp(from_unixtime(col("created_utc").cast("bigint"))),
        ).alias("review_date"),
        coalesce(col("channel"), col("channel_id"), lit("unknown")).alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        col("like_count").cast("int").alias("helpful_votes"),
        coalesce(col("url"), lit("")).alias("source_url"),
        coalesce(col("title"), lit("YouTube Review")).alias("display_name"),
        lit("Video Review").alias("display_category"),
        lit("video_transcript").alias("entity_type"),
    )
    return youtube_df, path


def normalize_reddit(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, REDDIT_FILE_CANDIDATES)
    if df is None:
        return None, None

    reddit_df = df.select(
        concat_ws("_", lit("reddit"), coalesce(col("source_id"), lit("unknown"))).alias("review_id"),
        coalesce(col("product_name"), lit("unknown")).alias("product_name"),
        coalesce(col("subreddit"), lit("general")).alias("product_category"),
        lit("reddit").alias("source"),
        lit(None).cast(DoubleType()).alias("rating_normalized"),
        when(
            col("title").isNotNull() & (trim(col("title")) != ""),
            concat_ws(". ", col("title"), coalesce(col("text"), lit(""))),
        ).otherwise(coalesce(col("text"), lit(""))).alias("review_text"),
        to_timestamp(from_unixtime(col("created_utc").cast("bigint"))).alias("review_date"),
        coalesce(col("author"), lit("unknown")).alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        col("score").cast("int").alias("helpful_votes"),
        coalesce(col("url"), lit("")).alias("source_url"),
        coalesce(col("title"), lit("Reddit Discussion")).alias("display_name"),
        coalesce(col("subreddit"), lit("general")).alias("display_category"),
        lit("forum_post").alias("entity_type"),
    )
    return reddit_df, path


def add_feature_columns(df: Any) -> Any:
    text = coalesce(col("review_text"), lit(""))
    normalized_text = trim(regexp_replace(text, r"\s+", " "))
    char_count = length(text)
    alpha_count = length(regexp_replace(text, "[^A-Za-z]", ""))
    uppercase_count = length(regexp_replace(text, "[^A-Z]", ""))
    exclamation_count = length(text) - length(regexp_replace(text, "!", ""))

    return (
        df
        .withColumn(
            "text_length_words",
            when(normalized_text == "", lit(0)).otherwise(size(split(normalized_text, " "))),
        )
        .withColumn("text_length_chars", char_count)
        .withColumn(
            "caps_ratio",
            when(alpha_count > 0, uppercase_count.cast(DoubleType()) / alpha_count.cast(DoubleType()))
            .otherwise(lit(0.0)),
        )
        .withColumn(
            "exclamation_ratio",
            when(char_count > 0, exclamation_count.cast(DoubleType()) / char_count.cast(DoubleType()))
            .otherwise(lit(0.0)),
        )
    )


def deduplicate_exact_reviews(df: Any) -> Any:
    before_count = df.count()
    deduplicated = df.dropDuplicates(["review_id"])
    removed = before_count - deduplicated.count()
    return deduplicated


def _publish_normalized_parquet(
    storage_manager: S3StorageManager,
    logger: logging.Logger,
    run_context: PipelineRunContext,
    output_path: Path,
) -> None:
    run_prefix = storage_manager.resolver.processed_run_prefix("normalized_parquet", run_context.run_id)
    current_prefix = storage_manager.resolver.processed_current_prefix("normalized_parquet")

    log_event(
        logger,
        "s3_upload_started",
        stage="normalized_parquet",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(output_path),
        output_path=run_prefix,
        status="started",
    )
    uploaded = storage_manager.upload_directory(output_path, run_prefix)
    log_event(
        logger,
        "s3_upload_completed",
        stage="normalized_parquet",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        input_path=str(output_path),
        output_path=run_prefix,
        file_count=len(uploaded),
        status="success",
    )

    promotion = storage_manager.promote_run_prefix(
        run_prefix,
        current_prefix,
        run_id=run_context.run_id,
        metadata={"stage": "normalized_parquet", "status": "success"},
    )
    log_event(
        logger,
        "latest_run_promoted",
        stage="normalized_parquet",
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        output_path=current_prefix,
        file_count=promotion["copied_count"],
        status="success",
    )


def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("spark.normalize_reviews")
    run_context = build_run_context(stage="normalize_reviews_spark")
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

    loaders = [
        normalize_amazon,
        normalize_yelp,
        normalize_ebay,
        normalize_ifixit,
        normalize_youtube,
        normalize_reddit,
    ]

    dataframes: list[Any] = []
    for loader in loaders:
        dataframe, source_path = loader(spark)
        if dataframe is not None:
            source_name = loader.__name__.replace("normalize_", "")
            source_count = dataframe.count()
            log_event(
                logger,
                "source_fetch_completed",
                source=source_name,
                stage=run_context.stage,
                run_id=run_context.run_id,
                dag_id=run_context.dag_id,
                task_id=run_context.task_id,
                input_path=str(source_path),
                record_count=source_count,
                status="success",
            )
            dataframes.append(dataframe)

    if not dataframes:
        spark.stop()
        log_event(
            logger,
            "normalization_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="failed",
            error_type="MissingInputError",
            error_message="No input source files were found for Spark normalization.",
        )
        raise RuntimeError("No input source files were found for Spark normalization.")

    unified_df = dataframes[0]
    for dataframe in dataframes[1:]:
        unified_df = unified_df.unionByName(dataframe)

    feature_started_at = time.perf_counter()
    log_event(
        logger,
        "feature_engineering_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        status="started",
    )
    unified_df = add_feature_columns(unified_df)
    log_event(
        logger,
        "feature_engineering_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        duration_ms=round((time.perf_counter() - feature_started_at) * 1000, 2),
        status="success",
    )

    dedup_started_at = time.perf_counter()
    pre_dedup_count = unified_df.count()
    log_event(
        logger,
        "dedup_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=pre_dedup_count,
        status="started",
    )
    unified_df = deduplicate_exact_reviews(unified_df)
    post_dedup_count = unified_df.count()
    log_event(
        logger,
        "dedup_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=post_dedup_count,
        duration_ms=round((time.perf_counter() - dedup_started_at) * 1000, 2),
        status="success",
    )

    if settings.normalized_parquet_path.exists():
        shutil.rmtree(settings.normalized_parquet_path)

    settings.normalized_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.write.mode("overwrite").parquet(str(settings.normalized_parquet_path))
    _publish_normalized_parquet(storage_manager, logger, run_context, settings.normalized_parquet_path)

    spark.stop()
    log_event(
        logger,
        "normalization_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=post_dedup_count,
        output_path=str(settings.normalized_parquet_path),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
    )


if __name__ == "__main__":
    main()
