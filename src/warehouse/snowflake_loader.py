"""Snowflake warehouse loader for ReviewPulse AI parquet outputs.

Run:
    poetry run python -m src.warehouse.snowflake_loader --dataset normalized
    poetry run python -m src.warehouse.snowflake_loader --dataset sentiment
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from src.common.run_context import build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import join_s3_uri
from src.common.structured_logging import configure_structured_logging, get_logger, log_event


DatasetName = Literal["normalized", "sentiment"]

NORMALIZED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("review_id", "VARCHAR"),
    ("product_name", "VARCHAR"),
    ("product_category", "VARCHAR"),
    ("source", "VARCHAR"),
    ("rating_normalized", "FLOAT"),
    ("review_text", "VARCHAR"),
    ("review_date", "TIMESTAMP_NTZ"),
    ("reviewer_id", "VARCHAR"),
    ("verified_purchase", "BOOLEAN"),
    ("helpful_votes", "NUMBER"),
    ("source_url", "VARCHAR"),
    ("display_name", "VARCHAR"),
    ("display_category", "VARCHAR"),
    ("entity_type", "VARCHAR"),
    ("text_length_words", "NUMBER"),
    ("text_length_chars", "NUMBER"),
    ("caps_ratio", "FLOAT"),
    ("exclamation_ratio", "FLOAT"),
)
SENTIMENT_COLUMNS: tuple[tuple[str, str], ...] = (
    *NORMALIZED_COLUMNS,
    ("aspect_labels", "VARCHAR"),
    ("aspect_count", "NUMBER"),
    ("aspect_details_json", "VARCHAR"),
    ("sentiment_label", "VARCHAR"),
    ("sentiment_score", "FLOAT"),
)


@dataclass(frozen=True, slots=True)
class SnowflakeDatasetSpec:
    name: DatasetName
    table_name: str
    stage_relative_prefix: str
    columns: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SnowflakeLoadConfig:
    database: str
    schema: str
    warehouse: str
    role: str
    stage_name: str
    file_format_name: str
    processed_stage_url: str
    storage_integration: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    truncate_before_load: bool
    datasets: dict[DatasetName, SnowflakeDatasetSpec]

    @classmethod
    def from_settings(cls, settings: Settings) -> "SnowflakeLoadConfig":
        if not settings.s3_enabled:
            raise RuntimeError("S3_BUCKET_NAME must be configured before loading Snowflake from S3.")
        if not settings.snowflake_enabled:
            raise RuntimeError("Snowflake account, user, password, warehouse, database, and schema are required.")

        processed_stage_url = join_s3_uri(
            settings.s3_bucket_name,
            settings.s3_processed_prefix,
            trailing_slash=True,
        )
        return cls(
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
            warehouse=settings.snowflake_warehouse,
            role=settings.snowflake_role,
            stage_name=settings.snowflake_stage,
            file_format_name=settings.snowflake_file_format,
            processed_stage_url=processed_stage_url,
            storage_integration=settings.snowflake_storage_integration,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_session_token=settings.aws_session_token,
            truncate_before_load=settings.snowflake_truncate_before_load,
            datasets={
                "normalized": SnowflakeDatasetSpec(
                    name="normalized",
                    table_name=settings.snowflake_normalized_table,
                    stage_relative_prefix="normalized_parquet/current/",
                    columns=NORMALIZED_COLUMNS,
                ),
                "sentiment": SnowflakeDatasetSpec(
                    name="sentiment",
                    table_name=settings.snowflake_sentiment_table,
                    stage_relative_prefix="sentiment_parquet/current/",
                    columns=SENTIMENT_COLUMNS,
                ),
            },
        )


def quote_identifier(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError("Snowflake identifier cannot be empty.")
    return '"' + cleaned.replace('"', '""') + '"'


def qualified_name(database: str, schema: str, object_name: str) -> str:
    return ".".join(
        quote_identifier(part)
        for part in (database, schema, object_name)
    )


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _stage_credentials_clause(config: SnowflakeLoadConfig) -> str:
    if config.storage_integration:
        return f"STORAGE_INTEGRATION = {quote_identifier(config.storage_integration)}"
    if not config.aws_access_key_id or not config.aws_secret_access_key:
        raise RuntimeError(
            "Set SNOWFLAKE_STORAGE_INTEGRATION or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "before creating the Snowflake S3 stage."
        )

    credential_parts = [
        f"AWS_KEY_ID = {sql_literal(config.aws_access_key_id)}",
        f"AWS_SECRET_KEY = {sql_literal(config.aws_secret_access_key)}",
    ]
    if config.aws_session_token:
        credential_parts.append(f"AWS_TOKEN = {sql_literal(config.aws_session_token)}")
    return "CREDENTIALS = (" + " ".join(credential_parts) + ")"


def build_context_sql(config: SnowflakeLoadConfig) -> list[str]:
    statements: list[str] = []
    if config.role:
        statements.append(f"USE ROLE {quote_identifier(config.role)}")
    statements.extend(
        [
            f"USE WAREHOUSE {quote_identifier(config.warehouse)}",
            f"USE DATABASE {quote_identifier(config.database)}",
            f"USE SCHEMA {quote_identifier(config.schema)}",
        ]
    )
    return statements


def build_file_format_sql(config: SnowflakeLoadConfig) -> str:
    file_format_name = qualified_name(config.database, config.schema, config.file_format_name)
    return (
        f"CREATE FILE FORMAT IF NOT EXISTS {file_format_name} "
        "TYPE = PARQUET USE_LOGICAL_TYPE = TRUE"
    )


def build_stage_sql(config: SnowflakeLoadConfig) -> str:
    stage_name = qualified_name(config.database, config.schema, config.stage_name)
    file_format_name = qualified_name(config.database, config.schema, config.file_format_name)
    credentials_clause = _stage_credentials_clause(config)
    return (
        f"CREATE OR REPLACE STAGE {stage_name} "
        f"URL = {sql_literal(config.processed_stage_url)} "
        f"{credentials_clause} "
        f"FILE_FORMAT = {file_format_name}"
    )


def build_create_table_sql(
    config: SnowflakeLoadConfig,
    dataset: SnowflakeDatasetSpec,
) -> str:
    table_name = qualified_name(config.database, config.schema, dataset.table_name)
    column_sql = ", ".join(
        f"{quote_identifier(column_name)} {column_type}"
        for column_name, column_type in dataset.columns
    )
    return f"CREATE TABLE IF NOT EXISTS {table_name} ({column_sql})"


def build_copy_sql(
    config: SnowflakeLoadConfig,
    dataset: SnowflakeDatasetSpec,
) -> str:
    table_name = qualified_name(config.database, config.schema, dataset.table_name)
    stage_name = qualified_name(config.database, config.schema, config.stage_name)
    file_format_name = qualified_name(config.database, config.schema, config.file_format_name)
    return (
        f"COPY INTO {table_name} "
        f"FROM @{stage_name}/{dataset.stage_relative_prefix} "
        "PATTERN = '.*[.]parquet$' "
        f"FILE_FORMAT = (FORMAT_NAME = {file_format_name}) "
        "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE "
        "ON_ERROR = ABORT_STATEMENT "
        "FORCE = TRUE"
    )


def build_count_sql(config: SnowflakeLoadConfig, dataset: SnowflakeDatasetSpec) -> str:
    table_name = qualified_name(config.database, config.schema, dataset.table_name)
    return f"SELECT COUNT(*) FROM {table_name}"


def build_truncate_sql(config: SnowflakeLoadConfig, dataset: SnowflakeDatasetSpec) -> str:
    table_name = qualified_name(config.database, config.schema, dataset.table_name)
    return f"TRUNCATE TABLE {table_name}"


def _require_snowflake_connector() -> Any:
    try:
        return importlib.import_module("snowflake.connector")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: snowflake-connector-python. "
            "Install project dependencies before running the Snowflake loader."
        ) from exc


def connect_to_snowflake(settings: Settings) -> Any:
    connector = _require_snowflake_connector()
    connect_kwargs: dict[str, str] = {
        "account": settings.snowflake_account,
        "user": settings.snowflake_user,
        "password": settings.snowflake_password,
        "warehouse": settings.snowflake_warehouse,
        "database": settings.snowflake_database,
        "schema": settings.snowflake_schema,
    }
    if settings.snowflake_role:
        connect_kwargs["role"] = settings.snowflake_role
    return connector.connect(**connect_kwargs)


def _execute_statement(connection: Any, sql: str) -> list[Any]:
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        fetchall = getattr(cursor, "fetchall", None)
        if callable(fetchall):
            return list(fetchall())
        return []
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def ensure_snowflake_objects(
    connection: Any,
    config: SnowflakeLoadConfig,
    datasets: list[SnowflakeDatasetSpec],
) -> None:
    statements = [
        *build_context_sql(config),
        build_file_format_sql(config),
        build_stage_sql(config),
        *[build_create_table_sql(config, dataset) for dataset in datasets],
    ]
    for statement in statements:
        _execute_statement(connection, statement)


def load_dataset_to_snowflake(
    connection: Any,
    config: SnowflakeLoadConfig,
    dataset: SnowflakeDatasetSpec,
) -> dict[str, Any]:
    if config.truncate_before_load:
        _execute_statement(connection, build_truncate_sql(config, dataset))

    copy_result = _execute_statement(connection, build_copy_sql(config, dataset))
    count_result = _execute_statement(connection, build_count_sql(config, dataset))
    row_count = int(count_result[0][0]) if count_result else 0
    return {
        "dataset": dataset.name,
        "table_name": dataset.table_name,
        "stage_prefix": dataset.stage_relative_prefix,
        "copy_result_rows": len(copy_result),
        "row_count": row_count,
    }


def resolve_dataset_names(selected: str) -> list[DatasetName]:
    if selected == "all":
        return ["normalized", "sentiment"]
    if selected in {"normalized", "sentiment"}:
        return [selected]  # type: ignore[list-item]
    raise ValueError(f"Unsupported Snowflake dataset: {selected}")


def load_snowflake_datasets(
    settings: Settings,
    *,
    selected_dataset: str = "normalized",
    connection: Any | None = None,
) -> list[dict[str, Any]]:
    config = SnowflakeLoadConfig.from_settings(settings)
    dataset_names = resolve_dataset_names(selected_dataset)
    datasets = [config.datasets[name] for name in dataset_names]

    owns_connection = connection is None
    active_connection = connection or connect_to_snowflake(settings)
    try:
        ensure_snowflake_objects(active_connection, config, datasets)
        return [
            load_dataset_to_snowflake(active_connection, config, dataset)
            for dataset in datasets
        ]
    finally:
        if owns_connection:
            close = getattr(active_connection, "close", None)
            if callable(close):
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load ReviewPulse parquet outputs into Snowflake.")
    parser.add_argument(
        "--dataset",
        choices=("normalized", "sentiment", "all"),
        default="normalized",
        help="Which processed parquet dataset to load from S3.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_structured_logging(settings.log_level)
    logger = get_logger("warehouse.snowflake")
    run_context = build_run_context(stage="load_snowflake")
    started_at = time.perf_counter()

    log_event(
        logger,
        "snowflake_load_started",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        selected_dataset=args.dataset,
        status="started",
    )
    try:
        results = load_snowflake_datasets(settings, selected_dataset=args.dataset)
    except Exception as exc:
        log_event(
            logger,
            "snowflake_load_failed",
            level=logging.ERROR,
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            selected_dataset=args.dataset,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise

    for result in results:
        log_event(
            logger,
            "snowflake_dataset_loaded",
            stage=run_context.stage,
            run_id=run_context.run_id,
            dag_id=run_context.dag_id,
            task_id=run_context.task_id,
            status="success",
            **result,
        )
    log_event(
        logger,
        "snowflake_load_completed",
        stage=run_context.stage,
        run_id=run_context.run_id,
        dag_id=run_context.dag_id,
        task_id=run_context.task_id,
        selected_dataset=args.dataset,
        dataset_count=len(results),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status="success",
    )


if __name__ == "__main__":
    main()
