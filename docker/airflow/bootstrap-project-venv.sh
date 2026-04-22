#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${REVIEWPULSE_PROJECT_ROOT:-/opt/reviewpulse}"
VENV_PATH="${REVIEWPULSE_PROJECT_VENV:-/opt/airflow/reviewpulse-venv}"

echo "Bootstrapping ReviewPulse project runtime into ${VENV_PATH}..."

python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --no-cache-dir --upgrade pip setuptools wheel

export POETRY_VIRTUALENVS_CREATE=false
export POETRY_REQUESTS_TIMEOUT="${POETRY_REQUESTS_TIMEOUT:-600}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"

cd "${PROJECT_ROOT}"
poetry install --no-root --only main
python -m pip install --no-cache-dir pytest

echo "ReviewPulse project runtime is ready at ${VENV_PATH}."
