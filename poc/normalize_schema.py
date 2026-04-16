"""
Compatibility wrapper for the shared normalization pipeline.

Run:
    poetry run python poc/normalize_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization.core import run_local_normalization


def run() -> list[dict]:
    return run_local_normalization()


if __name__ == "__main__":
    run()
