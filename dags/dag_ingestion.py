"""
ReviewPulse AI Airflow DAG for the ingestion and processing pipeline.

This DAG runs the checked-in pipeline stages in order:
1. Ingest Amazon raw data from the configured Hugging Face dataset.
2. Optionally ingest Yelp, eBay, iFixit, and YouTube raw data when their source configs are present.
3. Validate and normalize raw source data into a shared JSONL preview.
4. Run the Spark normalization job for the core sources.
5. Score sentiment.
6. Build embeddings.
7. Run tests.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class DagTaskSpec:
    task_id: str
    bash_command: str


def _project_python_invocation() -> str:
    project_python = os.getenv("REVIEWPULSE_PROJECT_PYTHON", "").strip()
    if project_python:
        return project_python
    return "poetry run python"


def _project_module_command(module_name: str) -> str:
    return f"cd {PROJECT_ROOT_POSIX} && {_project_python_invocation()} -m {module_name}"


def _project_pytest_command(test_target: str) -> str:
    return f"cd {PROJECT_ROOT_POSIX} && {_project_python_invocation()} -m pytest {test_target}"


def _has_values(values: tuple[str, ...]) -> bool:
    return any(str(value).strip() for value in values)


def _amazon_ingestion_enabled(settings: Any) -> bool:
    return bool(str(settings.amazon_dataset_name).strip() and str(settings.amazon_category).strip())


def _yelp_ingestion_enabled(settings: Any) -> bool:
    return bool(settings.has_yelp_dataset_source)


def _ebay_ingestion_enabled(settings: Any) -> bool:
    has_credentials = all(
        str(value).strip()
        for value in (settings.ebay_app_id, settings.ebay_dev_id, settings.ebay_cert_id)
    )
    has_workload = bool(
        settings.ebay_crawl_all_categories
        or _has_values(settings.ebay_category_ids)
        or _has_values(settings.ebay_search_queries)
    )
    return has_credentials and has_workload


def _ifixit_ingestion_enabled(settings: Any) -> bool:
    return bool(_has_values(settings.ifixit_guide_ids) or settings.ifixit_max_guides > 0)


def _youtube_ingestion_enabled(settings: Any) -> bool:
    return bool(
        str(settings.youtube_api_key).strip()
        or _has_values(settings.youtube_video_ids)
        or _has_values(settings.youtube_search_queries)
    )


def _build_source_task_specs(settings: Any) -> tuple[DagTaskSpec, ...]:
    task_specs: list[DagTaskSpec] = []

    if _amazon_ingestion_enabled(settings):
        task_specs.append(
            DagTaskSpec(
                task_id="ingest_amazon",
                bash_command=_project_module_command("src.ingestion.amazon"),
            )
        )
    if _yelp_ingestion_enabled(settings):
        task_specs.append(
            DagTaskSpec(
                task_id="ingest_yelp",
                bash_command=_project_module_command("src.ingestion.yelp"),
            )
        )
    if _ebay_ingestion_enabled(settings):
        task_specs.append(
            DagTaskSpec(
                task_id="ingest_ebay",
                bash_command=_project_module_command("src.ingestion.ebay"),
            )
        )
    if _ifixit_ingestion_enabled(settings):
        task_specs.append(
            DagTaskSpec(
                task_id="ingest_ifixit",
                bash_command=_project_module_command("src.ingestion.ifixit"),
            )
        )
    if _youtube_ingestion_enabled(settings):
        task_specs.append(
            DagTaskSpec(
                task_id="ingest_youtube",
                bash_command=_project_module_command("src.ingestion.youtube"),
            )
        )

    return tuple(task_specs)


def _build_processing_task_specs() -> tuple[DagTaskSpec, ...]:
    return (
        DagTaskSpec(
            task_id="normalize_local_preview",
            bash_command=_project_module_command("src.normalization.core"),
        ),
        DagTaskSpec(
            task_id="normalize_reviews_spark",
            bash_command=_project_module_command("src.spark.normalize_reviews_spark"),
        ),
        DagTaskSpec(
            task_id="score_sentiment",
            bash_command=_project_module_command("src.ml.sentiment_scoring"),
        ),
        DagTaskSpec(
            task_id="build_embeddings",
            bash_command=_project_module_command("src.retrieval.build_embeddings"),
        ),
        DagTaskSpec(
            task_id="run_tests",
            bash_command=_project_pytest_command("tests/test_normalization.py -v"),
        ),
    )


def _describe_dag_structure(settings: Any) -> str:
    source_task_specs = _build_source_task_specs(settings)

    lines = [
        "DAG: reviewpulse_pipeline",
        "Schedule: Daily at 2 AM UTC",
        "",
        "Task Flow:",
    ]
    if source_task_specs:
        lines.append("    Parallel source ingestion:")
        for spec in source_task_specs:
            lines.append(f"        {spec.task_id}")
        lines.append("            -> normalize_local_preview")
    else:
        lines.append("    normalize_local_preview")

    lines.extend(
        [
            "    normalize_local_preview",
            "        -> normalize_reviews_spark",
            "        -> score_sentiment",
            "        -> build_embeddings",
            "        -> run_tests",
        ]
    )
    return "\n".join(lines)


DAG_STRUCTURE = _describe_dag_structure(get_settings())


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
        source_tasks: dict[str, Any] = {}
        for spec in source_task_specs:
            source_tasks[spec.task_id] = BashOperator(
                task_id=spec.task_id,
                bash_command=spec.bash_command,
                env=task_env,
                on_execute_callback=_log_airflow_task_started,
                on_success_callback=_log_airflow_task_completed,
                on_failure_callback=_log_airflow_task_failed,
            )

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

        normalize_local_preview = processing_tasks["normalize_local_preview"]
        for task in source_tasks.values():
            task >> normalize_local_preview

        (
            processing_tasks["normalize_local_preview"]
            >> processing_tasks["normalize_reviews_spark"]
            >> processing_tasks["score_sentiment"]
            >> processing_tasks["build_embeddings"]
            >> processing_tasks["run_tests"]
        )

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
