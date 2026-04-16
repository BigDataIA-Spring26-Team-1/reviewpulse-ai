"""Project settings and path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    normalized_jsonl_path: Path
    normalized_parquet_path: Path
    sentiment_parquet_path: Path
    chroma_path: Path
    spark_master: str
    spark_sql_session_timezone: str


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

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        normalized_jsonl_path=(data_dir / "normalized_reviews.jsonl").resolve(),
        normalized_parquet_path=normalized_parquet_path,
        sentiment_parquet_path=sentiment_parquet_path,
        chroma_path=chroma_path,
        spark_master=os.getenv("SPARK_MASTER", "local[*]"),
        spark_sql_session_timezone=os.getenv("SPARK_SQL_SESSION_TIMEZONE", "UTC"),
    )
