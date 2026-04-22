"""
ReviewPulse AI Airflow DAG for the ingestion and processing pipeline.

This DAG runs the checked-in pipeline stages in order:
1. Optionally ingest Yelp raw data when YELP_DATASET_PATH or YELP_DATASET_S3_URI is configured.
2. Validate and normalize raw source data into a shared JSONL preview.
3. Run the Spark normalization job for the core sources.
4. Score sentiment.
5. Build embeddings.
6. Run tests.
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_POSIX = PROJECT_ROOT.as_posix()

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import get_settings
from src.common.structured_logging import configure_structured_logging, get_logger, log_event

DAG_STRUCTURE = """
DAG: reviewpulse_pipeline
Schedule: Daily at 2 AM UTC

Task Flow:
    ingest_yelp (optional; only when Yelp source config is provided)
        -> normalize_local_preview
    normalize_local_preview
        -> normalize_reviews_spark
        -> score_sentiment
        -> build_embeddings
        -> run_tests
"""


def _load_airflow_objects() -> tuple[Any, Any]:
    try:
        airflow_module = importlib.import_module("airflow")
        bash_module = importlib.import_module("airflow.operators.bash")
    except ImportError as exc:
        raise RuntimeError("Airflow is not installed.") from exc

    return airflow_module.DAG, bash_module.BashOperator


def _task_log_fields(context: dict[str, Any], status: str) -> dict[str, Any]:
    dag = context.get("dag")
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    fields: dict[str, Any] = {
        "dag_id": getattr(dag, "dag_id", None),
        "task_id": getattr(task_instance, "task_id", None),
        "run_id": context.get("run_id"),
        "stage": getattr(task_instance, "task_id", None),
        "status": status,
    }
    if exception is not None:
        fields["error_type"] = type(exception).__name__
        fields["error_message"] = str(exception)
    return fields


def _log_airflow_task_started(context: dict[str, Any]) -> None:
    configure_structured_logging("INFO", logger_name="reviewpulse.airflow")
    logger = get_logger("airflow")
    log_event(logger, "airflow_task_started", **_task_log_fields(context, "started"))


def _log_airflow_task_completed(context: dict[str, Any]) -> None:
    configure_structured_logging("INFO", logger_name="reviewpulse.airflow")
    logger = get_logger("airflow")
    log_event(logger, "airflow_task_completed", **_task_log_fields(context, "success"))


def _log_airflow_task_failed(context: dict[str, Any]) -> None:
    configure_structured_logging("INFO", logger_name="reviewpulse.airflow")
    logger = get_logger("airflow")
    log_event(
        logger,
        "airflow_task_failed",
        level=logging.ERROR,
        **_task_log_fields(context, "failed"),
    )


def _build_dag() -> Any:
    DAG, BashOperator = _load_airflow_objects()
    project_settings = get_settings()

    default_args = {
        "owner": "reviewpulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="reviewpulse_pipeline",
        default_args=default_args,
        description="ReviewPulse AI ingestion and processing pipeline",
        schedule="0 2 * * *",
        start_date=datetime(2026, 4, 1),
        catchup=False,
        tags=["reviewpulse", "pipeline"],
    ) as airflow_dag:
        task_env = {
            "REVIEWPULSE_RUN_ID": "{{ run_id }}",
            "REVIEWPULSE_DAG_ID": "{{ dag.dag_id }}",
            "REVIEWPULSE_TASK_ID": "{{ task.task_id }}",
        }

        ingest_yelp = None
        if project_settings.has_yelp_dataset_source:
            ingest_yelp = BashOperator(
                task_id="ingest_yelp",
                bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.ingestion.yelp",
                env=task_env,
                on_execute_callback=_log_airflow_task_started,
                on_success_callback=_log_airflow_task_completed,
                on_failure_callback=_log_airflow_task_failed,
            )

        normalize_local_preview = BashOperator(
            task_id="normalize_local_preview",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.normalization.core",
            env=task_env,
            on_execute_callback=_log_airflow_task_started,
            on_success_callback=_log_airflow_task_completed,
            on_failure_callback=_log_airflow_task_failed,
        )

        normalize_spark = BashOperator(
            task_id="normalize_reviews_spark",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.spark.normalize_reviews_spark",
            env=task_env,
            on_execute_callback=_log_airflow_task_started,
            on_success_callback=_log_airflow_task_completed,
            on_failure_callback=_log_airflow_task_failed,
        )

        score_sentiment = BashOperator(
            task_id="score_sentiment",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.ml.sentiment_scoring",
            env=task_env,
            on_execute_callback=_log_airflow_task_started,
            on_success_callback=_log_airflow_task_completed,
            on_failure_callback=_log_airflow_task_failed,
        )

        build_embeddings = BashOperator(
            task_id="build_embeddings",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.retrieval.build_embeddings",
            env=task_env,
            on_execute_callback=_log_airflow_task_started,
            on_success_callback=_log_airflow_task_completed,
            on_failure_callback=_log_airflow_task_failed,
        )

        run_tests = BashOperator(
            task_id="run_tests",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && pytest tests/test_normalization.py -v",
            env=task_env,
            on_execute_callback=_log_airflow_task_started,
            on_success_callback=_log_airflow_task_completed,
            on_failure_callback=_log_airflow_task_failed,
        )

        if ingest_yelp is not None:
            ingest_yelp >> normalize_local_preview

        normalize_local_preview >> normalize_spark >> score_sentiment >> build_embeddings >> run_tests

    return airflow_dag


dag: Any | None = None

try:
    dag = _build_dag()
except RuntimeError:
    dag = None


if __name__ == "__main__":
    if dag is None:
        configure_structured_logging("INFO", logger_name="reviewpulse.airflow")
        logger = get_logger("airflow")
        log_event(
            logger,
            "airflow_task_failed",
            level=logging.ERROR,
            dag_id="reviewpulse_pipeline",
            task_id="dag_bootstrap",
            stage="dag_bootstrap",
            status="failed",
            error_type="MissingDependency",
            error_message="Airflow is not installed. This file documents the DAG structure.",
        )
    else:
        configure_structured_logging("INFO", logger_name="reviewpulse.airflow")
        logger = get_logger("airflow")
        log_event(
            logger,
            "airflow_task_completed",
            dag_id=getattr(dag, "dag_id", "reviewpulse_pipeline"),
            task_id="dag_bootstrap",
            stage="dag_bootstrap",
            status="success",
        )
