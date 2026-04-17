from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


try:
    import mlflow
except ModuleNotFoundError:  # pragma: no cover - fallback for bare environments
    mlflow = None


def is_mlflow_available() -> bool:
    return mlflow is not None


def configure_mlflow(tracking_uri: str | None = None, experiment_name: str = "ecommerce-recsys") -> None:
    if mlflow is None:
        return
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextlib.contextmanager
def optional_mlflow_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
    if mlflow is None:
        yield
        return
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield


def log_params(params: dict) -> None:
    if mlflow is not None:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if mlflow is not None:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    if mlflow is not None:
        mlflow.log_artifact(path)
