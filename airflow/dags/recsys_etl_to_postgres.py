from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


ROOT = Path(os.getenv("RECSYS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PYTHON_BIN = os.getenv("RECSYS_PYTHON_BIN", sys.executable)
ARTIFACTS_DIR = ROOT / "artifacts"
VALIDATION_DAYS = os.getenv("RECSYS_VALIDATION_DAYS", "14")
STAGING_SCHEMA = os.getenv("RECSYS_STAGING_SCHEMA", "recsys_staging")
MART_SCHEMA = os.getenv("RECSYS_MART_SCHEMA", "recsys_mart")
CONN_ID = os.getenv("RECSYS_AIRFLOW_CONN_ID", "recsys_postgres")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def prepare_runtime_dirs() -> None:
    for relative in ["artifacts/data", "artifacts/reports"]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def transform_local_data() -> None:
    run_command(
        [
            PYTHON_BIN,
            "scripts/run_etl.py",
            "transform",
            "--data-dir",
            str(ROOT),
            "--artifacts-dir",
            str(ARTIFACTS_DIR),
            "--validation-days",
            VALIDATION_DAYS,
            "--run-id",
            "airflow-etl",
        ]
    )


def load_transformed_to_postgres() -> None:
    run_command(
        [
            PYTHON_BIN,
            "scripts/run_etl.py",
            "load",
            "--artifacts-dir",
            str(ARTIFACTS_DIR),
            "--staging-schema",
            STAGING_SCHEMA,
            "--mart-schema",
            MART_SCHEMA,
            "--run-id",
            "airflow-etl",
            "--conn-id",
            CONN_ID,
        ]
    )


def validate_loaded_tables() -> None:
    run_command(
        [
            PYTHON_BIN,
            "scripts/run_etl.py",
            "validate",
            "--artifacts-dir",
            str(ARTIFACTS_DIR),
            "--staging-schema",
            STAGING_SCHEMA,
            "--mart-schema",
            MART_SCHEMA,
            "--conn-id",
            CONN_ID,
        ]
    )


with DAG(
    dag_id="ecommerce_recommender_etl_to_postgres",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["recsys", "etl", "postgres"],
) as dag:
    prepare_task = PythonOperator(task_id="prepare_runtime_dirs", python_callable=prepare_runtime_dirs)
    transform_task = PythonOperator(task_id="transform_local_data", python_callable=transform_local_data)
    load_task = PythonOperator(task_id="load_transformed_to_postgres", python_callable=load_transformed_to_postgres)
    validate_task = PythonOperator(task_id="validate_loaded_tables", python_callable=validate_loaded_tables)

    prepare_task >> transform_task >> load_task >> validate_task
