from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


ROOT = Path(os.getenv("RECSYS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PYTHON_BIN = os.getenv("RECSYS_PYTHON_BIN", sys.executable)


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


def rebuild_notebooks() -> None:
    run_command([PYTHON_BIN, "scripts/build_notebooks.py"])


with DAG(
    dag_id="ecommerce_recommender_retrain",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["recsys", "mlflow", "airflow"],
) as dag:
    prepare_task = PythonOperator(task_id="prepare_runtime_dirs", python_callable=prepare_runtime_dirs)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    notebooks_task = PythonOperator(task_id="rebuild_notebooks", python_callable=rebuild_notebooks)

    prepare_task >> train_task >> notebooks_task
