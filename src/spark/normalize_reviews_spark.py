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
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


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
    row_number = pyspark_functions.row_number
    DoubleType = pyspark_types.DoubleType
    Window = importlib.import_module("pyspark.sql.window").Window
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
    row_number = _missing_pyspark
    DoubleType = _missing_pyspark
    Window = _missing_pyspark


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.common.run_context import PipelineRunContext, build_run_context
from src.common.spark_runtime import ensure_local_hadoop_home
from src.normalization.core import (
    AMAZON_FILE_CANDIDATES,
    EBAY_FILE_CANDIDATES,
    IFIXIT_FILE_CANDIDATES,
    REDDIT_FILE_CANDIDATES,
    YOUTUBE_FILE_CANDIDATES,
    resolve_source_input_path,
    resolve_yelp_source_paths,
    stage_s3_raw_current_sources,
)


def build_spark() -> Any:
    if not PYSPARK_AVAILABLE:
        raise RuntimeError("Missing dependency: `pyspark`. " + DEPENDENCY_SETUP_HINT)
    settings = get_settings()
    hadoop_home = ensure_local_hadoop_home(PROJECT_ROOT)
    builder = (
        SparkSession.builder
        .appName("ReviewPulse-Normalization")
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", settings.spark_sql_session_timezone)
    )
    for env_name, spark_key in (
        ("SPARK_DRIVER_MEMORY", "spark.driver.memory"),
        ("SPARK_EXECUTOR_MEMORY", "spark.executor.memory"),
        ("SPARK_DRIVER_MAX_RESULT_SIZE", "spark.driver.maxResultSize"),
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            builder = builder.config(spark_key, value)

    spark_local_dir = _resolve_spark_local_dir(os.getenv("SPARK_LOCAL_DIR", "").strip())
    builder = builder.config("spark.local.dir", str(spark_local_dir))

    shuffle_partitions = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "").strip() or "800"
    builder = (
        builder
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.adaptive.enabled", "true")
    )
    if hadoop_home is not None:
        builder = builder.config("spark.hadoop.hadoop.home.dir", str(hadoop_home))
    return builder.getOrCreate()


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


def _optional_column(df: Any, column_name: str, *, cast_type: str | None = None) -> Any:
    expression = col(column_name) if column_name in df.columns else lit(None)
    if cast_type is not None:
        expression = expression.cast(cast_type)
    return expression


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / ".reviewpulse-write-test"
        probe_path.write_text("", encoding="utf-8")
        probe_path.unlink()
        return True
    except OSError:
        return False


def _resolve_spark_local_dir(configured_path: str) -> Path:
    candidates: list[Path] = []
    if configured_path:
        configured = Path(configured_path)
        candidates.append(configured if configured.is_absolute() else (PROJECT_ROOT / configured).resolve())
    candidates.append((PROJECT_ROOT / ".runtime" / "spark-local").resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if _is_writable_directory(resolved_candidate):
            return resolved_candidate

    raise RuntimeError("Unable to determine a writable Spark local directory.")


def normalize_amazon(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, AMAZON_FILE_CANDIDATES)
    if df is None:
        return None, None

    url_column = _optional_column(df, "url", cast_type="string")

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
            url_column.isNotNull() & (trim(url_column) != ""),
            url_column,
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
    settings = get_settings()
    review_path, business_path = resolve_yelp_source_paths(settings.data_dir)
    if review_path is None:
        return None, None

    reviews = spark.read.json(str(review_path))
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

    item_id = coalesce(_optional_column(df, "item_id", cast_type="string"), lit("unknown"))
    title = coalesce(
        _optional_column(df, "title", cast_type="string"),
        _optional_column(df, "item_title", cast_type="string"),
        item_id,
    )
    category = coalesce(
        _optional_column(df, "category", cast_type="string"),
        _optional_column(df, "primary_category", cast_type="string"),
        lit("unknown"),
    )
    review_text = coalesce(
        _optional_column(df, "feedback_text", cast_type="string"),
        _optional_column(df, "review_text", cast_type="string"),
        lit(""),
    )
    review_date = to_timestamp(
        coalesce(
            _optional_column(df, "listing_date", cast_type="string"),
            _optional_column(df, "review_date", cast_type="string"),
        )
    )
    reviewer_id = coalesce(
        _optional_column(df, "seller_id", cast_type="string"),
        _optional_column(df, "seller", cast_type="string"),
        _optional_column(df, "reviewer_id", cast_type="string"),
        lit("unknown"),
    )
    helpful_votes = coalesce(
        _optional_column(df, "feedback_count", cast_type="int"),
        _optional_column(df, "helpful_votes", cast_type="int"),
    )
    url_column = _optional_column(df, "url", cast_type="string")

    ebay_df = df.select(
        concat_ws("_", lit("ebay"), item_id).alias("review_id"),
        title.alias("product_name"),
        category.alias("product_category"),
        lit("ebay").alias("source"),
        _scale_column("seller_rating", 0.0, 100.0).alias("rating_normalized"),
        review_text.alias("review_text"),
        review_date.alias("review_date"),
        reviewer_id.alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        helpful_votes.alias("helpful_votes"),
        when(
            url_column.isNotNull() & (trim(url_column) != ""),
            url_column,
        ).otherwise(concat_ws("", lit("https://www.ebay.com/itm/"), item_id)).alias("source_url"),
        title.alias("display_name"),
        coalesce(category, lit("Marketplace Listing")).alias("display_category"),
        lit("listing_review").alias("entity_type"),
    )
    return ebay_df, path


def normalize_ifixit(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, IFIXIT_FILE_CANDIDATES)
    if df is None:
        return None, None

    guide_id = coalesce(
        _optional_column(df, "guide_id", cast_type="string"),
        _optional_column(df, "source_id", cast_type="string"),
        lit("unknown"),
    )
    product_name = coalesce(
        _optional_column(df, "device_name", cast_type="string"),
        _optional_column(df, "title", cast_type="string"),
        _optional_column(df, "guide_id", cast_type="string"),
        lit("unknown"),
    )
    display_name = coalesce(
        _optional_column(df, "title", cast_type="string"),
        _optional_column(df, "device_name", cast_type="string"),
        _optional_column(df, "guide_id", cast_type="string"),
        lit("unknown"),
    )
    product_category = coalesce(
        _optional_column(df, "device_category", cast_type="string"),
        lit("repairability"),
    )
    display_category = coalesce(
        _optional_column(df, "device_category", cast_type="string"),
        lit("Repair Guide"),
    )
    review_text = coalesce(
        _optional_column(df, "review_text", cast_type="string"),
        _optional_column(df, "text", cast_type="string"),
        lit(""),
    )
    published_date = _optional_column(df, "published_date", cast_type="string")
    review_date = coalesce(
        when(
            published_date.rlike(r"^\d+$"),
            to_timestamp(from_unixtime(_optional_column(df, "published_date", cast_type="bigint"))),
        ).otherwise(to_timestamp(published_date)),
        lit(None).cast("timestamp"),
    )
    reviewer_id = coalesce(
        _optional_column(df, "author", cast_type="string"),
        lit("unknown"),
    )
    helpful_votes = _optional_column(df, "helpful_votes", cast_type="int")
    url_column = _optional_column(df, "url", cast_type="string")

    ifixit_df = df.select(
        concat_ws("_", lit("ifixit"), guide_id).alias("review_id"),
        product_name.alias("product_name"),
        product_category.alias("product_category"),
        lit("ifixit").alias("source"),
        _scale_column("repairability_score", 1.0, 10.0).alias("rating_normalized"),
        review_text.alias("review_text"),
        review_date.alias("review_date"),
        reviewer_id.alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        helpful_votes.alias("helpful_votes"),
        when(
            url_column.isNotNull() & (trim(url_column) != ""),
            url_column,
        ).otherwise(concat_ws("", lit("https://www.ifixit.com/Guide/"), guide_id)).alias("source_url"),
        display_name.alias("display_name"),
        display_category.alias("display_category"),
        lit("repair_review").alias("entity_type"),
    )
    return ifixit_df, path


def normalize_youtube(spark: Any) -> tuple[Any | None, Path | None]:
    df, path = _read_json(spark, YOUTUBE_FILE_CANDIDATES)
    if df is None:
        return None, None

    source_id = coalesce(
        _optional_column(df, "source_id", cast_type="string"),
        _optional_column(df, "video_id", cast_type="string"),
        lit("unknown"),
    )
    review_text = coalesce(
        _optional_column(df, "text", cast_type="string"),
        _optional_column(df, "transcript", cast_type="string"),
        lit(""),
    )
    review_date = coalesce(
        to_timestamp(_optional_column(df, "published_date", cast_type="string")),
        to_timestamp(from_unixtime(_optional_column(df, "created_utc", cast_type="bigint"))),
    )
    reviewer_id = coalesce(
        _optional_column(df, "channel", cast_type="string"),
        _optional_column(df, "channel_id", cast_type="string"),
        lit("unknown"),
    )

    youtube_df = df.select(
        concat_ws("_", lit("youtube"), source_id).alias("review_id"),
        coalesce(_optional_column(df, "product_name", cast_type="string"), lit("unknown")).alias("product_name"),
        coalesce(_optional_column(df, "product_category", cast_type="string"), lit("unknown")).alias("product_category"),
        lit("youtube").alias("source"),
        lit(None).cast(DoubleType()).alias("rating_normalized"),
        review_text.alias("review_text"),
        review_date.alias("review_date"),
        reviewer_id.alias("reviewer_id"),
        lit(None).cast("boolean").alias("verified_purchase"),
        _optional_column(df, "like_count", cast_type="int").alias("helpful_votes"),
        coalesce(_optional_column(df, "url", cast_type="string"), lit("")).alias("source_url"),
        coalesce(_optional_column(df, "title", cast_type="string"), lit("YouTube Review")).alias("display_name"),
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
    dedup_window = Window.partitionBy("review_id").orderBy(col("review_date").desc_nulls_last())
    return (
        df
        .repartition(col("review_id"))
        .withColumn("_review_rank", row_number().over(dedup_window))
        .filter(col("_review_rank") == 1)
        .drop("_review_rank")
    )


def _arrow_type_for_spark_type(type_name: str) -> pa.DataType:
    mapping = {
        "string": pa.string(),
        "double": pa.float64(),
        "float": pa.float32(),
        "long": pa.int64(),
        "integer": pa.int32(),
        "int": pa.int32(),
        "short": pa.int16(),
        "byte": pa.int8(),
        "boolean": pa.bool_(),
        "timestamp": pa.timestamp("us"),
        "date": pa.date32(),
    }
    return mapping.get(type_name, pa.string())


def _arrow_schema_from_spark_schema(schema: Any) -> pa.Schema:
    return pa.schema(
        [
            pa.field(field.name, _arrow_type_for_spark_type(field.dataType.typeName()), nullable=True)
            for field in schema.fields
        ]
    )


def _write_parquet_via_arrow(
    df: Any,
    output_path: Path,
    *,
    logger: logging.Logger,
    run_context: PipelineRunContext,
    batch_size: int = 50_000,
) -> int:
    output_path = output_path.resolve()
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    part_index = 0
    total_records = 0
    buffered_rows: list[dict[str, Any]] = []
    schema = _arrow_schema_from_spark_schema(df.schema)

    def flush_batch(rows: list[dict[str, Any]]) -> None:
        nonlocal part_index, total_records
        if not rows:
            return

        table = pa.Table.from_pylist(rows, schema=schema)
        part_path = output_path / f"part-{part_index:05d}.parquet"
        pq.write_table(table, part_path, compression="snappy")
        part_index += 1
        total_records += len(rows)
        log_event(
            logger,
            "parquet_write_progress",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            output_path=str(output_path),
            file_count=part_index,
            record_count=total_records,
            status="running",
        )
        rows.clear()

    iterator = df.toLocalIterator()
    for row in iterator:
        buffered_rows.append(row.asDict(recursive=True))
        if len(buffered_rows) >= batch_size:
            flush_batch(buffered_rows)

    flush_batch(buffered_rows)
    if total_records <= 0:
        raise RuntimeError("Spark normalization produced zero rows for parquet output.")
    return total_records


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
    stage_s3_raw_current_sources(
        settings=settings,
        storage_manager=storage_manager,
        logger=logger,
        run_context=run_context,
    )
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
        removed_count=pre_dedup_count - post_dedup_count,
    )

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

    if settings.normalized_parquet_path.exists():
        shutil.rmtree(settings.normalized_parquet_path)

    settings.normalized_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        written_record_count = _write_parquet_via_arrow(
            unified_df,
            settings.normalized_parquet_path,
            logger=logger,
            run_context=run_context,
        )
    else:
        unified_df.write.mode("overwrite").parquet(str(settings.normalized_parquet_path))
        written_record_count = post_dedup_count
    _publish_normalized_parquet(storage_manager, logger, run_context, settings.normalized_parquet_path)

    spark.stop()
    log_event(
        logger,
        "normalization_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        record_count=written_record_count,
        output_path=str(settings.normalized_parquet_path),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
    )


if __name__ == "__main__":
    main()
