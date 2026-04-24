"""
ReviewPulse AI Airflow DAG for normalization-only reruns.

This DAG assumes the raw source files have already been ingested and focuses on
rebuilding normalized and downstream artifacts:
1. Rebuild the unified local JSONL preview from the latest raw files.
2. Rebuild Spark normalized parquet output.
3. Recompute sentiment labels and scores.
4. Load Snowflake when configured.
5. Rebuild embeddings.
6. Re-run normalization tests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from dags.dag_ingestion import (
    _build_processing_task_specs,
    _load_airflow_objects,
    _log_airflow_task_completed,
    _log_airflow_task_failed,
    _log_airflow_task_started,
)
from src.common.settings import get_settings
from src.common.structured_logging import configure_structured_logging, get_logger, log_event


DAG_STRUCTURE = """
DAG: reviewpulse_reprocess_pipeline
Schedule: Manual / on-demand

Task Flow:
    normalize_local_preview
        -> normalize_reviews_spark
        -> score_sentiment
        -> load_snowflake (when configured)
        -> build_embeddings
        -> run_tests
"""


def _build_dag() -> Any:
    DAG, BashOperator = _load_airflow_objects()
    processing_task_specs = _build_processing_task_specs(get_settings())

    default_args = {
        "owner": "reviewpulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="reviewpulse_reprocess_pipeline",
        default_args=default_args,
        description="ReviewPulse AI processing-only rerun pipeline",
        schedule=None,
        start_date=datetime(2026, 4, 1),
        catchup=False,
        tags=["reviewpulse", "reprocess"],
    ) as airflow_dag:
        task_env = {
            "REVIEWPULSE_RUN_ID": "{{ run_id }}",
            "REVIEWPULSE_DAG_ID": "{{ dag.dag_id }}",
            "REVIEWPULSE_TASK_ID": "{{ task.task_id }}",
        }

        processing_tasks: dict[str, Any] = {}
        for spec in processing_task_specs:
            processing_tasks[spec.task_id] = BashOperator(
                task_id=spec.task_id,
                bash_command=spec.bash_command,
                env=task_env,
                on_execute_callback=_log_airflow_task_started,
                on_success_callback=_log_airflow_task_completed,
                on_failure_callback=_log_airflow_task_failed,
            )

        for upstream_spec, downstream_spec in zip(processing_task_specs, processing_task_specs[1:]):
            processing_tasks[upstream_spec.task_id] >> processing_tasks[downstream_spec.task_id]

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
            dag_id="reviewpulse_reprocess_pipeline",
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
            dag_id=getattr(dag, "dag_id", "reviewpulse_reprocess_pipeline"),
            task_id="dag_bootstrap",
            stage="dag_bootstrap",
            status="success",
        )
