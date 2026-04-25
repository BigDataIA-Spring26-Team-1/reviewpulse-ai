"""
ReviewPulse AI Airflow DAG for the ingestion and processing pipeline.

This DAG runs the checked-in pipeline stages in order:
1. Ingest enabled raw sources.
2. Validate and normalize raw source data into a shared JSONL preview.
3. Run the Spark normalization job for the core sources.
4. Score sentiment.
5. Load Snowflake when configured.
6. Build embeddings.
7. Run tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


from dags.reviewpulse_airflow_common import (
    TaskSpec,
    _build_processing_task_specs,
    _load_airflow_objects,
    _log_airflow_task_completed,
    _log_airflow_task_failed,
    _log_airflow_task_started,
    _module_command,
)
from src.common.settings import Settings, get_settings
from src.common.structured_logging import configure_structured_logging, get_logger, log_event

DAG_STRUCTURE = """
DAG: reviewpulse_pipeline
Schedule: Daily at 2 AM UTC

Task Flow:
    ingest_enabled_sources
        -> normalize_local_preview
        -> normalize_reviews_spark
        -> score_sentiment
        -> load_snowflake (when configured)
        -> build_embeddings
        -> run_tests
"""


def _build_source_task_specs(settings: Settings) -> list[TaskSpec]:
    specs = [
        TaskSpec(
            task_id="ingest_amazon",
            bash_command=_module_command("src.ingestion.amazon"),
        )
    ]

    if settings.has_yelp_dataset_source:
        specs.append(
            TaskSpec(
                task_id="ingest_yelp",
                bash_command=_module_command("src.ingestion.yelp"),
            )
        )

    if settings.ebay_app_id and settings.ebay_cert_id and (
        settings.ebay_search_queries or settings.ebay_category_ids
    ):
        specs.append(
            TaskSpec(
                task_id="ingest_ebay",
                bash_command=_module_command("src.ingestion.ebay"),
            )
        )

    if settings.ifixit_guide_ids:
        specs.append(
            TaskSpec(
                task_id="ingest_ifixit",
                bash_command=_module_command("src.ingestion.ifixit"),
            )
        )

    if settings.youtube_video_ids or settings.youtube_search_queries:
        specs.append(
            TaskSpec(
                task_id="ingest_youtube",
                bash_command=_module_command("src.ingestion.youtube"),
            )
        )

    return specs


def _build_dag() -> Any:
    DAG, BashOperator = _load_airflow_objects()
    project_settings = get_settings()
    source_task_specs = _build_source_task_specs(project_settings)
    processing_task_specs = _build_processing_task_specs(project_settings)

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

        airflow_tasks: dict[str, Any] = {}
        for spec in [*source_task_specs, *processing_task_specs]:
            airflow_tasks[spec.task_id] = BashOperator(
                task_id=spec.task_id,
                bash_command=spec.bash_command,
                env=task_env,
                on_execute_callback=_log_airflow_task_started,
                on_success_callback=_log_airflow_task_completed,
                on_failure_callback=_log_airflow_task_failed,
            )

        normalize_local_preview = airflow_tasks["normalize_local_preview"]
        for spec in source_task_specs:
            airflow_tasks[spec.task_id] >> normalize_local_preview

        for upstream_spec, downstream_spec in zip(processing_task_specs, processing_task_specs[1:]):
            airflow_tasks[upstream_spec.task_id] >> airflow_tasks[downstream_spec.task_id]

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
