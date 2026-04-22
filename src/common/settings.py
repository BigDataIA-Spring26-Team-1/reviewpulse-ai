"""Project settings and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_project_path(value: str, default: str) -> Path:
    raw_value = value or default
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_optional_project_path(value: str) -> Optional[Path]:
    raw_value = value.strip()
    if not raw_value:
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    values = [part.strip() for part in raw_value.replace("\n", ",").split(",")]
    return tuple(value for value in values if value)


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    app_name: str
    app_env: str
    log_level: str
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    normalized_jsonl_path: Path
    normalized_parquet_path: Path
    sentiment_parquet_path: Path
    chroma_path: Path
    aws_region: str
    s3_bucket_name: str
    s3_raw_prefix: str
    s3_processed_prefix: str
    spark_master: str
    spark_sql_session_timezone: str
    huggingface_token: str
    amazon_dataset_name: str
    amazon_category: str
    amazon_batch_size: int
    amazon_max_records: int
    yelp_dataset_path: Path | None
    yelp_dataset_s3_uri: str | None
    ebay_app_id: str
    ebay_dev_id: str
    ebay_cert_id: str
    ebay_site_id: str
    ebay_marketplace_id: str
    ebay_search_queries: tuple[str, ...]
    ebay_max_items_per_query: int
    ifixit_base_url: str
    ifixit_guide_ids: tuple[str, ...]
    youtube_api_key: str
    youtube_video_ids: tuple[str, ...]
    youtube_transcript_languages: tuple[str, ...]

    @property
    def s3_enabled(self) -> bool:
        return bool(self.s3_bucket_name.strip())

    @property
    def yelp_dataset_source(self) -> str | Path | None:
        if self.yelp_dataset_s3_uri:
            return self.yelp_dataset_s3_uri
        return self.yelp_dataset_path

    @property
    def has_yelp_dataset_source(self) -> bool:
        return self.yelp_dataset_source is not None


def get_settings() -> Settings:
    data_dir = _resolve_project_path(os.getenv("DATA_DIR", "./data"), "./data")
    raw_data_dir = _resolve_project_path(os.getenv("RAW_DATA_DIR", "./data/raw"), "./data/raw")
    processed_data_dir = _resolve_project_path(
        os.getenv("PROCESSED_DATA_DIR", "./data/processed"),
        "./data/processed",
    )
    normalized_parquet_path = _resolve_project_path(
        os.getenv("PARQUET_INPUT_PATH", "./data/normalized_reviews_parquet"),
        "./data/normalized_reviews_parquet",
    )
    sentiment_parquet_path = _resolve_project_path(
        os.getenv("PARQUET_OUTPUT_PATH", "./data/reviews_with_sentiment_parquet"),
        "./data/reviews_with_sentiment_parquet",
    )
    chroma_path = _resolve_project_path(
        os.getenv("CHROMA_PATH", "./data/chromadb_reviews"),
        "./data/chromadb_reviews",
    )
    yelp_dataset_raw = os.getenv("YELP_DATASET_PATH", "").strip()
    yelp_dataset_s3_uri = os.getenv("YELP_DATASET_S3_URI", "").strip()
    if yelp_dataset_raw.startswith("s3://") and not yelp_dataset_s3_uri:
        yelp_dataset_s3_uri = yelp_dataset_raw
        yelp_dataset_path = None
    else:
        yelp_dataset_path = _resolve_optional_project_path(yelp_dataset_raw)

    return Settings(
        project_root=PROJECT_ROOT,
        app_name=os.getenv("APP_NAME", "ReviewPulse AI"),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        normalized_jsonl_path=(data_dir / "normalized_reviews.jsonl").resolve(),
        normalized_parquet_path=normalized_parquet_path,
        sentiment_parquet_path=sentiment_parquet_path,
        chroma_path=chroma_path,
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        s3_bucket_name=os.getenv("S3_BUCKET_NAME", "").strip(),
        s3_raw_prefix=os.getenv("S3_RAW_PREFIX", "raw"),
        s3_processed_prefix=os.getenv("S3_PROCESSED_PREFIX", "processed"),
        spark_master=os.getenv("SPARK_MASTER", "local[*]"),
        spark_sql_session_timezone=os.getenv("SPARK_SQL_SESSION_TIMEZONE", "UTC"),
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN", "").strip(),
        amazon_dataset_name=os.getenv("AMAZON_DATASET_NAME", "McAuley-Lab/Amazon-Reviews-2023").strip(),
        amazon_category=os.getenv("AMAZON_CATEGORY", "Electronics").strip(),
        amazon_batch_size=_int_env("AMAZON_BATCH_SIZE", 10000),
        amazon_max_records=_int_env("AMAZON_MAX_RECORDS", 0),
        yelp_dataset_path=yelp_dataset_path,
        yelp_dataset_s3_uri=yelp_dataset_s3_uri or None,
        ebay_app_id=os.getenv("EBAY_APP_ID", "").strip(),
        ebay_dev_id=os.getenv("EBAY_DEV_ID", "").strip(),
        ebay_cert_id=os.getenv("EBAY_CERT_ID", "").strip(),
        ebay_site_id=os.getenv("EBAY_SITE_ID", "0").strip(),
        ebay_marketplace_id=os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US").strip(),
        ebay_search_queries=_csv_env("EBAY_SEARCH_QUERIES"),
        ebay_max_items_per_query=_int_env("EBAY_MAX_ITEMS_PER_QUERY", 50),
        ifixit_base_url=os.getenv("IFIXIT_BASE_URL", "https://www.ifixit.com").strip(),
        ifixit_guide_ids=_csv_env("IFIXIT_GUIDE_IDS"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        youtube_video_ids=_csv_env("YOUTUBE_VIDEO_IDS"),
        youtube_transcript_languages=_csv_env("YOUTUBE_TRANSCRIPT_LANGUAGES") or ("en",),
    )
