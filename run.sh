#!/bin/bash
set -e
cd "$(dirname "$0")"

case "$1" in
  backend)
    source .venv/bin/activate
    uvicorn src.api.main:app --reload --reload-dir src
    ;;
  frontend)
    source .venv/bin/activate
    streamlit run src/frontend/app.py
    ;;
  health)
    curl -s http://127.0.0.1:8000/health | python3 -m json.tool
    ;;
  health-detailed)
    curl -s http://127.0.0.1:8000/health/detailed | python3 -m json.tool
    ;;
  *)
    echo "Usage: ./run.sh [backend|frontend|health|health-detailed]"
    ;;
esac
