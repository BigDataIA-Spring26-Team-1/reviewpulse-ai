from __future__ import annotations

import shutil

from src.warehouse.snowflake_loader import (
    SnowflakeLoadConfig,
    build_stage_sql,
    load_snowflake_datasets,
)
from tests.test_source_ingestion import build_test_settings, make_workspace


class FakeCursor:
    def __init__(self, connection: "FakeSnowflakeConnection") -> None:
        self.connection = connection
        self.statement = ""

    def execute(self, statement: str) -> None:
        self.statement = statement
        self.connection.statements.append(statement)

    def fetchall(self) -> list[tuple[int]]:
        if self.statement.startswith("SELECT COUNT(*)"):
            return [(self.connection.count_result,)]
        if self.statement.startswith("COPY INTO"):
            return [(1,), (2,)]
        return []

    def close(self) -> None:
        self.connection.closed_cursors += 1


class FakeSnowflakeConnection:
    def __init__(self, *, count_result: int = 8_148_920) -> None:
        self.count_result = count_result
        self.statements: list[str] = []
        self.closed = False
        self.closed_cursors = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        self.closed = True


def test_stage_sql_uses_s3_credentials_when_storage_integration_missing():
    workspace = make_workspace("snowflake_stage_credentials")
    try:
        settings = build_test_settings(workspace)
        config = SnowflakeLoadConfig.from_settings(settings)

        stage_sql = build_stage_sql(config)

        assert "CREATE OR REPLACE STAGE" in stage_sql
        assert "URL = 's3://reviewpulse-bucket/processed/'" in stage_sql
        assert "CREDENTIALS = (" in stage_sql
        assert "AWS_KEY_ID = 'aws-key'" in stage_sql
        assert "AWS_SECRET_KEY = 'aws-secret'" in stage_sql
        assert "STORAGE_INTEGRATION" not in stage_sql
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_stage_sql_prefers_storage_integration_over_raw_aws_keys():
    workspace = make_workspace("snowflake_stage_integration")
    try:
        settings = build_test_settings(workspace)
        config = SnowflakeLoadConfig.from_settings(settings)
        config = SnowflakeLoadConfig(
            database=config.database,
            schema=config.schema,
            warehouse=config.warehouse,
            role=config.role,
            stage_name=config.stage_name,
            file_format_name=config.file_format_name,
            processed_stage_url=config.processed_stage_url,
            storage_integration="REVIEWPULSE_STORAGE_INTEGRATION",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            truncate_before_load=config.truncate_before_load,
            datasets=config.datasets,
        )

        stage_sql = build_stage_sql(config)

        assert 'STORAGE_INTEGRATION = "REVIEWPULSE_STORAGE_INTEGRATION"' in stage_sql
        assert "AWS_KEY_ID" not in stage_sql
        assert "AWS_SECRET_KEY" not in stage_sql
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_load_snowflake_datasets_loads_normalized_prefix_only():
    workspace = make_workspace("snowflake_load_normalized")
    try:
        settings = build_test_settings(workspace)
        connection = FakeSnowflakeConnection()

        results = load_snowflake_datasets(
            settings,
            selected_dataset="normalized",
            connection=connection,
        )

        copy_statements = [
            statement
            for statement in connection.statements
            if statement.startswith("COPY INTO")
        ]
        assert len(copy_statements) == 1
        assert "normalized_parquet/current/" in copy_statements[0]
        assert "sentiment_parquet/current/" not in copy_statements[0]
        assert "PATTERN = '.*[.]parquet$'" in copy_statements[0]
        assert any(statement.startswith("TRUNCATE TABLE") for statement in connection.statements)
        assert results == [
            {
                "dataset": "normalized",
                "table_name": "NORMALIZED_REVIEWS",
                "stage_prefix": "normalized_parquet/current/",
                "copy_result_rows": 2,
                "row_count": 8_148_920,
            }
        ]
        assert not connection.closed
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
