FROM apache/airflow:3.2.1-python3.11

USER airflow
ENV PATH="/home/airflow/poetry-venv/bin:${PATH}" \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/reviewpulse \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/reviewpulse/dags

WORKDIR /opt/reviewpulse

RUN python -m venv /home/airflow/poetry-venv \
    && /home/airflow/poetry-venv/bin/pip install --no-cache-dir poetry==2.1.2

COPY --chown=airflow:0 dags ./dags
COPY --chown=airflow:0 src ./src
COPY --chown=airflow:0 tests ./tests
COPY --chown=airflow:0 docker/airflow ./docker/airflow
COPY --chown=airflow:0 pyproject.toml poetry.lock README.md .env.example ./

RUN chmod +x /opt/reviewpulse/docker/airflow/bootstrap-project-venv.sh \
    && chmod +x /opt/reviewpulse/docker/airflow/run-airflow-standalone.sh
