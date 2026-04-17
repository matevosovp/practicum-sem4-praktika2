from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg import sql
from psycopg.types.json import Jsonb

from .serialization import load_json, save_json


def build_reporting_payload(root_dir: Path) -> dict[str, object]:
    reports_dir = root_dir / "artifacts" / "reports"
    data_dir = root_dir / "artifacts" / "data"

    evaluation_summary = load_json(reports_dir / "evaluation_summary.json")
    etl_summary = load_json(reports_dir / "etl_summary.json")
    eda_summary = load_json(reports_dir / "eda_runtime_summary.json")

    payload: dict[str, object] = {
        "best_model_name": evaluation_summary.get("best_model_name"),
        "top_k": evaluation_summary.get("top_k"),
        "cutoff_timestamp": evaluation_summary.get("cutoff_timestamp"),
        "model_metrics": evaluation_summary.get("all_metrics", {}),
        "etl_summary": etl_summary,
        "eda_runtime_summary": eda_summary,
    }

    evaluation_frame_path = data_dir / "evaluation_frame.parquet"
    if evaluation_frame_path.exists():
        evaluation_frame = pd.read_parquet(evaluation_frame_path)
        payload["evaluation_dataset_summary"] = {
            "rows": int(len(evaluation_frame)),
            "users": int(evaluation_frame["visitorid"].nunique()) if not evaluation_frame.empty else 0,
            "avg_history_events": float(round(evaluation_frame["history_events"].mean(), 4)) if not evaluation_frame.empty else 0.0,
        }

    aggregated_train_path = data_dir / "aggregated_train.parquet"
    if aggregated_train_path.exists():
        aggregated_train = pd.read_parquet(aggregated_train_path, columns=["visitorid", "itemid"])
        payload["training_dataset_summary"] = {
            "rows": int(len(aggregated_train)),
            "users": int(aggregated_train["visitorid"].nunique()) if not aggregated_train.empty else 0,
            "items": int(aggregated_train["itemid"].nunique()) if not aggregated_train.empty else 0,
        }

    return payload


def write_reporting_payload(root_dir: Path, payload: dict[str, object]) -> Path:
    path = root_dir / "artifacts" / "reports" / "reporting_metrics.json"
    save_json(payload, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_registry(root_dir: Path, run_id: str) -> pd.DataFrame:
    tracked_files = [
        ("model_bundle", root_dir / "models" / "recommender.joblib", "model"),
        ("serving_history", root_dir / "models" / "serving_history.parquet", "model"),
        ("evaluation_summary", root_dir / "artifacts" / "reports" / "evaluation_summary.json", "report"),
        ("reporting_metrics", root_dir / "artifacts" / "reports" / "reporting_metrics.json", "report"),
        ("etl_summary", root_dir / "artifacts" / "reports" / "etl_summary.json", "report"),
    ]
    rows = []
    loaded_at = datetime.now(timezone.utc).isoformat()
    for artifact_name, artifact_path, artifact_type in tracked_files:
        if not artifact_path.exists():
            continue
        stat = artifact_path.stat()
        rows.append(
            {
                "run_id": run_id,
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "artifact_path": str(artifact_path.relative_to(root_dir)),
                "file_size_bytes": int(stat.st_size),
                "sha256": _sha256(artifact_path),
                "created_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "registered_at_utc": loaded_at,
            }
        )
    return pd.DataFrame(rows)


def load_reporting_to_db(
    conn,
    mart_schema: str,
    run_id: str,
    payload: dict[str, object],
    artifact_registry: pd.DataFrame,
) -> dict[str, int]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    model_metrics_rows = [
        (
            run_id,
            model_name,
            float(metrics.get("recall_at_k", 0.0)),
            float(metrics.get("hit_rate_at_k", 0.0)),
            float(metrics.get("ndcg_at_k", 0.0)),
            float(metrics.get("evaluated_users", 0.0)),
            float(metrics.get("catalog_coverage", 0.0)),
        )
        for model_name, metrics in payload.get("model_metrics", {}).items()
    ]
    artifact_rows = [
        (
            row.run_id,
            row.artifact_name,
            row.artifact_type,
            row.artifact_path,
            int(row.file_size_bytes),
            row.sha256,
            row.created_at_utc,
            row.registered_at_utc,
        )
        for row in artifact_registry.itertuples(index=False)
    ]

    with conn.cursor() as cursor:
        cursor.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(mart_schema))
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.reporting_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    best_model_name TEXT,
                    top_k BIGINT,
                    cutoff_timestamp BIGINT,
                    etl_summary_json JSONB,
                    eda_runtime_summary_json JSONB,
                    evaluation_dataset_summary_json JSONB,
                    training_dataset_summary_json JSONB
                )
                """
            ).format(sql.Identifier(mart_schema))
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.model_metrics_history (
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    recall_at_k DOUBLE PRECISION,
                    hit_rate_at_k DOUBLE PRECISION,
                    ndcg_at_k DOUBLE PRECISION,
                    evaluated_users DOUBLE PRECISION,
                    catalog_coverage DOUBLE PRECISION
                )
                """
            ).format(sql.Identifier(mart_schema))
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.artifact_registry (
                    run_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    file_size_bytes BIGINT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    registered_at_utc TEXT NOT NULL
                )
                """
            ).format(sql.Identifier(mart_schema))
        )

        cursor.execute(
            sql.SQL(
                """
                INSERT INTO {}.reporting_runs (
                    run_id, created_at_utc, best_model_name, top_k, cutoff_timestamp,
                    etl_summary_json, eda_runtime_summary_json,
                    evaluation_dataset_summary_json, training_dataset_summary_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    created_at_utc = EXCLUDED.created_at_utc,
                    best_model_name = EXCLUDED.best_model_name,
                    top_k = EXCLUDED.top_k,
                    cutoff_timestamp = EXCLUDED.cutoff_timestamp,
                    etl_summary_json = EXCLUDED.etl_summary_json,
                    eda_runtime_summary_json = EXCLUDED.eda_runtime_summary_json,
                    evaluation_dataset_summary_json = EXCLUDED.evaluation_dataset_summary_json,
                    training_dataset_summary_json = EXCLUDED.training_dataset_summary_json
                """
            ).format(sql.Identifier(mart_schema)),
            (
                run_id,
                created_at_utc,
                payload.get("best_model_name"),
                payload.get("top_k"),
                payload.get("cutoff_timestamp"),
                Jsonb(payload.get("etl_summary", {})),
                Jsonb(payload.get("eda_runtime_summary", {})),
                Jsonb(payload.get("evaluation_dataset_summary", {})),
                Jsonb(payload.get("training_dataset_summary", {})),
            ),
        )

        if model_metrics_rows:
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {}.model_metrics_history (
                        run_id, model_name, recall_at_k, hit_rate_at_k, ndcg_at_k, evaluated_users, catalog_coverage
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(sql.Identifier(mart_schema)),
                model_metrics_rows,
            )
        if artifact_rows:
            cursor.executemany(
                sql.SQL(
                    """
                    INSERT INTO {}.artifact_registry (
                        run_id, artifact_name, artifact_type, artifact_path, file_size_bytes,
                        sha256, created_at_utc, registered_at_utc
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(sql.Identifier(mart_schema)),
                artifact_rows,
            )
    conn.commit()

    return {
        f"{mart_schema}.reporting_runs": 1,
        f"{mart_schema}.model_metrics_history": int(len(model_metrics_rows)),
        f"{mart_schema}.artifact_registry": int(len(artifact_rows)),
    }
