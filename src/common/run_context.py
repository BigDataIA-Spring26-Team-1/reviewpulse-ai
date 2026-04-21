"""Run context helpers for batch and Airflow-compatible pipeline execution."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


RUN_ID_ENV_VARS = (
    "REVIEWPULSE_RUN_ID",
    "AIRFLOW_CTX_RUN_ID",
    "AIRFLOW_CTX_DAG_RUN_ID",
)
DAG_ID_ENV_VARS = ("REVIEWPULSE_DAG_ID", "AIRFLOW_CTX_DAG_ID")
TASK_ID_ENV_VARS = ("REVIEWPULSE_TASK_ID", "AIRFLOW_CTX_TASK_ID")


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = cleaned.strip("-.")
    return cleaned[:120] or "unknown"


def generate_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    return f"run_{timestamp}_{token}"


def _first_env_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id:
        return sanitize_identifier(run_id)

    env_value = _first_env_value(RUN_ID_ENV_VARS)
    if env_value:
        return sanitize_identifier(env_value)

    return generate_run_id()


@dataclass(frozen=True, slots=True)
class PipelineRunContext:
    run_id: str
    stage: str
    source: str | None = None
    dag_id: str | None = None
    task_id: str | None = None

    def as_log_fields(self) -> dict[str, str]:
        fields: dict[str, str] = {
            "run_id": self.run_id,
            "stage": self.stage,
        }
        if self.source:
            fields["source"] = self.source
        if self.dag_id:
            fields["dag_id"] = self.dag_id
        if self.task_id:
            fields["task_id"] = self.task_id
        return fields


def build_run_context(
    stage: str,
    source: str | None = None,
    run_id: str | None = None,
    dag_id: str | None = None,
    task_id: str | None = None,
) -> PipelineRunContext:
    raw_dag_id = dag_id or _first_env_value(DAG_ID_ENV_VARS)
    raw_task_id = task_id or _first_env_value(TASK_ID_ENV_VARS)
    resolved_dag_id = sanitize_identifier(raw_dag_id) if raw_dag_id else None
    resolved_task_id = sanitize_identifier(raw_task_id) if raw_task_id else None
    resolved_source = sanitize_identifier(source) if source else None

    return PipelineRunContext(
        run_id=resolve_run_id(run_id),
        stage=sanitize_identifier(stage),
        source=resolved_source,
        dag_id=resolved_dag_id,
        task_id=resolved_task_id,
    )
