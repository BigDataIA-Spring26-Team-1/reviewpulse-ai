"""Shared helpers for ReviewPulse Airflow DAG definitions."""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_POSIX = PROJECT_ROOT.as_posix()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.settings import Settings
from src.common.structured_logging import configure_structured_logging, get_logger, log_event


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


def _build_processing_task_specs(settings: Settings | None = None) -> list[TaskSpec]:
    specs = [
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
    ]
    if settings is not None and settings.s3_enabled and settings.snowflake_enabled:
        specs.append(
            TaskSpec(
                task_id="load_snowflake",
                bash_command=_module_command("src.warehouse.snowflake_loader --dataset all"),
            )
        )

    specs.extend(
        [
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
    )
    return specs
