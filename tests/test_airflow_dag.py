from __future__ import annotations

import shutil
from dataclasses import replace

import dags.dag_ingestion as dag_ingestion
import dags.dag_reprocessing as dag_reprocessing
from tests.test_source_ingestion import build_test_settings, make_workspace


class FakeDAG:
    _stack: list["FakeDAG"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.task_dict: dict[str, FakeBashOperator] = {}

    def __enter__(self) -> "FakeDAG":
        self.__class__._stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.__class__._stack.pop()


class FakeBashOperator:
    def __init__(
        self,
        *,
        task_id: str,
        bash_command: str,
        env: dict[str, str],
        on_execute_callback=None,
        on_success_callback=None,
        on_failure_callback=None,
    ) -> None:
        self.task_id = task_id
        self.bash_command = bash_command
        self.env = env
        self.on_execute_callback = on_execute_callback
        self.on_success_callback = on_success_callback
        self.on_failure_callback = on_failure_callback
        self.upstream_task_ids: set[str] = set()
        self.downstream_task_ids: set[str] = set()
        FakeDAG._stack[-1].task_dict[task_id] = self

    def __rshift__(self, other: "FakeBashOperator") -> "FakeBashOperator":
        self.downstream_task_ids.add(other.task_id)
        other.upstream_task_ids.add(self.task_id)
        return other


def test_build_source_task_specs_includes_all_configured_sources():
    workspace = make_workspace("airflow_source_specs_full")
    try:
        settings = replace(
            build_test_settings(workspace),
            yelp_dataset_path=workspace,
        )

        specs = dag_ingestion._build_source_task_specs(settings)

        assert [spec.task_id for spec in specs] == [
            "ingest_amazon",
            "ingest_yelp",
            "ingest_ebay",
            "ingest_ifixit",
            "ingest_youtube",
        ]
        assert specs[0].bash_command.endswith("poetry run python -m src.ingestion.amazon")
        assert specs[-1].bash_command.endswith("poetry run python -m src.ingestion.youtube")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_build_source_task_specs_skips_unconfigured_optional_sources():
    workspace = make_workspace("airflow_source_specs_minimal")
    try:
        settings = replace(
            build_test_settings(workspace),
            yelp_dataset_path=None,
            yelp_dataset_s3_uri=None,
            ebay_app_id="",
            ebay_dev_id="",
            ebay_cert_id="",
            ebay_search_queries=tuple(),
            ebay_category_ids=tuple(),
            ebay_crawl_all_categories=False,
            ifixit_guide_ids=tuple(),
            ifixit_max_guides=0,
            youtube_api_key="",
            youtube_video_ids=tuple(),
            youtube_search_queries=tuple(),
        )

        specs = dag_ingestion._build_source_task_specs(settings)

        assert [spec.task_id for spec in specs] == ["ingest_amazon"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_build_dag_wires_all_enabled_source_tasks(monkeypatch):
    workspace = make_workspace("airflow_build_dag")
    try:
        settings = replace(
            build_test_settings(workspace),
            yelp_dataset_path=workspace,
        )

        monkeypatch.setattr(dag_ingestion, "_load_airflow_objects", lambda: (FakeDAG, FakeBashOperator))
        monkeypatch.setattr(dag_ingestion, "get_settings", lambda: settings)

        airflow_dag = dag_ingestion._build_dag()

        assert airflow_dag.kwargs["dag_id"] == "reviewpulse_pipeline"
        assert sorted(airflow_dag.task_dict) == [
            "build_embeddings",
            "ingest_amazon",
            "ingest_ebay",
            "ingest_ifixit",
            "ingest_yelp",
            "ingest_youtube",
            "load_snowflake",
            "normalize_local_preview",
            "normalize_reviews_spark",
            "run_tests",
            "score_sentiment",
        ]

        normalize_preview = airflow_dag.task_dict["normalize_local_preview"]
        assert normalize_preview.upstream_task_ids == {
            "ingest_amazon",
            "ingest_yelp",
            "ingest_ebay",
            "ingest_ifixit",
            "ingest_youtube",
        }
        assert airflow_dag.task_dict["normalize_reviews_spark"].upstream_task_ids == {"normalize_local_preview"}
        assert airflow_dag.task_dict["score_sentiment"].upstream_task_ids == {"normalize_reviews_spark"}
        assert airflow_dag.task_dict["load_snowflake"].upstream_task_ids == {"score_sentiment"}
        assert airflow_dag.task_dict["build_embeddings"].upstream_task_ids == {"load_snowflake"}
        assert airflow_dag.task_dict["run_tests"].upstream_task_ids == {"build_embeddings"}
        assert airflow_dag.task_dict["ingest_ebay"].bash_command.endswith(
            "poetry run python -m src.ingestion.ebay"
        )
        assert airflow_dag.task_dict["run_tests"].bash_command.endswith(
            "poetry run python -m pytest tests/test_normalization.py -v"
        )
        assert airflow_dag.task_dict["ingest_amazon"].env["REVIEWPULSE_RUN_ID"] == "{{ run_id }}"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_reprocessing_dag_wires_processing_chain(monkeypatch):
    workspace = make_workspace("airflow_reprocessing_dag")
    try:
        settings = build_test_settings(workspace)
        monkeypatch.setattr(dag_reprocessing, "_load_airflow_objects", lambda: (FakeDAG, FakeBashOperator))
        monkeypatch.setattr(dag_reprocessing, "get_settings", lambda: settings)

        airflow_dag = dag_reprocessing._build_dag()

        assert airflow_dag.kwargs["dag_id"] == "reviewpulse_reprocess_pipeline"
        assert airflow_dag.kwargs["schedule"] is None
        assert sorted(airflow_dag.task_dict) == [
            "build_embeddings",
            "load_snowflake",
            "normalize_local_preview",
            "normalize_reviews_spark",
            "run_tests",
            "score_sentiment",
        ]
        assert airflow_dag.task_dict["normalize_local_preview"].upstream_task_ids == set()
        assert airflow_dag.task_dict["normalize_reviews_spark"].upstream_task_ids == {"normalize_local_preview"}
        assert airflow_dag.task_dict["score_sentiment"].upstream_task_ids == {"normalize_reviews_spark"}
        assert airflow_dag.task_dict["load_snowflake"].upstream_task_ids == {"score_sentiment"}
        assert airflow_dag.task_dict["build_embeddings"].upstream_task_ids == {"load_snowflake"}
        assert airflow_dag.task_dict["run_tests"].upstream_task_ids == {"build_embeddings"}
        assert airflow_dag.task_dict["run_tests"].bash_command.endswith(
            "poetry run python -m pytest tests/test_normalization.py -v"
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
