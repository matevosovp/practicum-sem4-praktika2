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
CONN_ID = os.getenv("RECSYS_AIRFLOW_CONN_ID", "recsys_postgres")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def prepare_runtime_dirs() -> None:
    for relative in ["artifacts/data", "artifacts/reports", "models"]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def train_model() -> None:
    run_command(
        [
            PYTHON_BIN,
            "scripts/train_model.py",
            "--data-dir",
            str(ROOT),
            "--artifacts-dir",
            str(ROOT / "artifacts"),
            "--models-dir",
            str(ROOT / "models"),
            "--run-name",
            "airflow-retrain",
        ]
    )


def export_reporting_metrics() -> None:
    run_command(
        [
            PYTHON_BIN,
            "scripts/export_reporting_metrics.py",
            "--run-id",
            "airflow-retrain",
            "--mart-schema",
            os.getenv("RECSYS_MART_SCHEMA", "recsys_mart"),
            "--persist-db",
            "--conn-id",
            CONN_ID,
        ]
    )


with DAG(
    dag_id="ecommerce_recommender_retrain",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["recsys", "mlflow", "airflow"],
) as dag:
    prepare_task = PythonOperator(task_id="prepare_runtime_dirs", python_callable=prepare_runtime_dirs)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    reporting_task = PythonOperator(task_id="export_reporting_metrics", python_callable=export_reporting_metrics)

    prepare_task >> train_task >> reporting_task
