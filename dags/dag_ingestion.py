"""
ReviewPulse AI Airflow DAG for the ingestion and processing pipeline.

This DAG runs the checked-in pipeline stages in order:
1. Ingest enabled raw sources.
2. Validate and normalize raw source data into a shared JSONL preview.
3. Run the Spark normalization job for the core sources.
4. Score sentiment.
5. Build embeddings.
6. Run tests.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_POSIX = PROJECT_ROOT.as_posix()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        -> build_embeddings
        -> run_tests
"""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    bash_command: str


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


def _module_command(module_path: str) -> str:
    return f"cd {PROJECT_ROOT_POSIX} && poetry run python -m {module_path}"


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

    if settings.ebay_app_id and settings.ebay_cert_id and settings.ebay_search_queries:
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


def _build_processing_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id="normalize_local_preview",
            bash_command=_module_command("src.normalization.core"),
        ),
        TaskSpec(
            task_id="normalize_reviews_spark",
            bash_command=_module_command("src.spark.normalize_reviews_spark"),
        ),
        TaskSpec(
            task_id="score_sentiment",
            bash_command=_module_command("src.ml.sentiment_scoring"),
        ),
        TaskSpec(
            task_id="build_embeddings",
            bash_command=_module_command("src.retrieval.build_embeddings"),
        ),
        TaskSpec(
            task_id="run_tests",
            bash_command=(
                f"cd {PROJECT_ROOT_POSIX} && "
                "poetry run python -m pytest tests/test_normalization.py -v"
            ),
        ),
    ]


def _build_dag() -> Any:
    DAG, BashOperator = _load_airflow_objects()
    project_settings = get_settings()
    source_task_specs = _build_source_task_specs(project_settings)
    processing_task_specs = _build_processing_task_specs()

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
