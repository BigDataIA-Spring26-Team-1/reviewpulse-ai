"""
ReviewPulse AI Airflow DAG for the implemented MVP pipeline.

This DAG follows the proposal's implementation order:
1. Validate and normalize raw source data into a shared JSONL preview.
2. Run the Spark normalization job for the core sources.
3. Score sentiment.
4. Build embeddings.
5. Run tests.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_POSIX = PROJECT_ROOT.as_posix()

DAG_STRUCTURE = """
DAG: reviewpulse_mvp_pipeline
Schedule: Daily at 2 AM UTC

Task Flow:
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


def _build_dag() -> Any:
    DAG, BashOperator = _load_airflow_objects()

    default_args = {
        "owner": "reviewpulse",
        "depends_on_past": False,
        "email_on_failure": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="reviewpulse_mvp_pipeline",
        default_args=default_args,
        description="Proposal-aligned MVP pipeline for ReviewPulse AI",
        schedule="0 2 * * *",
        start_date=datetime(2026, 4, 1),
        catchup=False,
        tags=["reviewpulse", "mvp"],
    ) as airflow_dag:
        normalize_local_preview = BashOperator(
            task_id="normalize_local_preview",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.normalization.core",
        )

        normalize_spark = BashOperator(
            task_id="normalize_reviews_spark",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.spark.normalize_reviews_spark",
        )

        score_sentiment = BashOperator(
            task_id="score_sentiment",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.ml.sentiment_scoring",
        )

        build_embeddings = BashOperator(
            task_id="build_embeddings",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && python -m src.retrieval.build_embeddings",
        )

        run_tests = BashOperator(
            task_id="run_tests",
            bash_command=f"cd {PROJECT_ROOT_POSIX} && pytest tests/test_normalization.py -v",
        )

        normalize_local_preview >> normalize_spark >> score_sentiment >> build_embeddings >> run_tests

    return airflow_dag


dag: Any | None = None

try:
    dag = _build_dag()
except RuntimeError:
    dag = None


if __name__ == "__main__":
    if dag is None:
        print("Airflow is not installed. This file documents the DAG structure.")
        print(DAG_STRUCTURE)
    else:
        print(f"Loaded DAG: {getattr(dag, 'dag_id', 'reviewpulse_mvp_pipeline')}")
