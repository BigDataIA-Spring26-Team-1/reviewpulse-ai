from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.common import spark_runtime


def test_ensure_local_spark_dir_uses_configured_directory(monkeypatch):
    workspace = Path(__file__).resolve().parent / "_tmp_spark_runtime_local_dir"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        configured_dir = workspace / "spark-local-configured"
        monkeypatch.setenv("SPARK_LOCAL_DIR", str(configured_dir))
        monkeypatch.delenv("SPARK_LOCAL_DIRS", raising=False)

        spark_local_dir = spark_runtime.ensure_local_spark_dir(workspace)

        assert spark_local_dir == configured_dir.resolve()
        assert os.environ["SPARK_LOCAL_DIR"] == str(configured_dir.resolve())
        assert os.environ["SPARK_LOCAL_DIRS"] == str(configured_dir.resolve())
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_ensure_local_spark_dir_falls_back_to_workspace_directory(monkeypatch):
    workspace = Path(__file__).resolve().parent / "_tmp_spark_runtime_local_dir_fallback"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        blocked_dir = (workspace / "blocked").resolve()
        fallback_dir = (workspace / ".runtime" / "spark-local").resolve()
        monkeypatch.setenv("SPARK_LOCAL_DIR", str(blocked_dir))
        monkeypatch.delenv("SPARK_LOCAL_DIRS", raising=False)

        original = spark_runtime._ensure_writable_directory

        def fake_ensure_writable_directory(path: Path) -> bool:
            if path == blocked_dir:
                return False
            return original(path)

        monkeypatch.setattr(spark_runtime, "_ensure_writable_directory", fake_ensure_writable_directory)

        spark_local_dir = spark_runtime.ensure_local_spark_dir(workspace)

        assert spark_local_dir == fallback_dir
        assert os.environ["SPARK_LOCAL_DIR"] == str(fallback_dir)
        assert os.environ["SPARK_LOCAL_DIRS"] == str(fallback_dir)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_ensure_local_hadoop_home_compiles_and_sets_env(monkeypatch):
    workspace = Path(__file__).resolve().parent / "_tmp_spark_runtime"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        compiler_path = workspace / "csc.exe"
        compiler_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(spark_runtime.os, "name", "nt", raising=False)
        monkeypatch.delenv("HADOOP_HOME", raising=False)
        monkeypatch.delenv("hadoop.home.dir", raising=False)
        monkeypatch.setattr(spark_runtime, "_find_csc_executable", lambda: compiler_path)

        def fake_run(*args, **kwargs):
            command = args[0] if args else kwargs["args"]
            output_arg = next(arg for arg in command if str(arg).startswith("/out:"))
            output_path = Path(str(output_arg).split(":", 1)[1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"MZ")
            return None

        monkeypatch.setattr(spark_runtime.subprocess, "run", fake_run)

        hadoop_home = spark_runtime.ensure_local_hadoop_home(workspace)

        assert hadoop_home == (workspace / ".runtime" / "hadoop").resolve()
        assert os.environ["HADOOP_HOME"] == str(hadoop_home)
        assert os.environ["hadoop.home.dir"] == str(hadoop_home)
        assert (hadoop_home / "bin" / "winutils.exe").exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
