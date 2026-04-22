from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.common import spark_runtime


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
