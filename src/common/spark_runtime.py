"""Spark runtime helpers for local Windows execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


WINUTILS_STUB_SOURCE = """using System;

public static class WinUtilsStub
{
    public static int Main(string[] args)
    {
        return 0;
    }
}
"""


def _candidate_csc_paths() -> tuple[Path, ...]:
    return (
        Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
        Path(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe"),
        Path(r"C:\Windows\Microsoft.NET\Framework64\v3.5\csc.exe"),
        Path(r"C:\Windows\Microsoft.NET\Framework\v3.5\csc.exe"),
    )


def _find_csc_executable() -> Path | None:
    for path in _candidate_csc_paths():
        if path.exists():
            return path
    return None


def _compile_winutils_stub(output_path: Path) -> None:
    compiler_path = _find_csc_executable()
    if compiler_path is None:
        raise RuntimeError(
            "Spark on Windows requires HADOOP_HOME/winutils.exe to write local files. "
            "No C# compiler was found to generate the local winutils shim."
        )

    source_path = output_path.with_suffix(".cs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(WINUTILS_STUB_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            str(compiler_path),
            "/nologo",
            "/target:exe",
            f"/out:{output_path}",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not output_path.exists():
        raise RuntimeError(f"winutils shim compilation did not produce {output_path}")


def ensure_local_hadoop_home(project_root: Path) -> Path | None:
    if os.name != "nt":
        return None

    configured_home = os.getenv("HADOOP_HOME", "").strip() or os.getenv("hadoop.home.dir", "").strip()
    if configured_home:
        configured_path = Path(configured_home).resolve()
        if (configured_path / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(configured_path)
            os.environ["hadoop.home.dir"] = str(configured_path)
            return configured_path

    hadoop_home = (project_root / ".runtime" / "hadoop").resolve()
    winutils_path = hadoop_home / "bin" / "winutils.exe"
    if not winutils_path.exists():
        _compile_winutils_stub(winutils_path)

    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["hadoop.home.dir"] = str(hadoop_home)

    current_path = os.environ.get("PATH", "")
    bin_path = str(winutils_path.parent)
    if not current_path:
        os.environ["PATH"] = bin_path
    elif bin_path not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_path + os.pathsep + current_path

    return hadoop_home
