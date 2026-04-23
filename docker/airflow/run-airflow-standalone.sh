#!/usr/bin/env bash
set -euo pipefail

AIRFLOW_HOME="${AIRFLOW_HOME:-/opt/airflow}"
PROJECT_VENV="${REVIEWPULSE_PROJECT_VENV:-${AIRFLOW_HOME}/reviewpulse-venv}"
PASSWORDS_FILE="${AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE:-${AIRFLOW_HOME}/simple_auth_manager_passwords.json.generated}"
ADMIN_USERNAME="${AIRFLOW_ADMIN_USERNAME:-admin}"
ADMIN_PASSWORD="${AIRFLOW_ADMIN_PASSWORD:-}"

mkdir -p "${AIRFLOW_HOME}"

if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
  /opt/reviewpulse/docker/airflow/bootstrap-project-venv.sh
fi

if [[ -z "${ADMIN_PASSWORD}" ]]; then
  echo "AIRFLOW_ADMIN_PASSWORD must be set before starting deployed Airflow." >&2
  exit 1
fi

export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS="${ADMIN_USERNAME}:admin"
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE="${PASSWORDS_FILE}"
export REVIEWPULSE_PROJECT_PYTHON="${PROJECT_VENV}/bin/python"

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {os.environ["AIRFLOW_ADMIN_USERNAME"]: os.environ["AIRFLOW_ADMIN_PASSWORD"]}
path.write_text(json.dumps(payload), encoding="utf-8")
PY

exec airflow standalone
