from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, MetaData, Table, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import BIGINT, FLOAT, INTEGER, JSON, SMALLINT, TEXT

from .config import ETLConfig, TrainingConfig
from .data import ensure_dir, read_parquet
from .db import ensure_schema
from .pipeline import prepare_training_frames
from .serialization import load_json
from .serialization import save_json


def run_local_transform(config: ETLConfig) -> dict[str, str | int]:
    training_config = TrainingConfig(
        data_dir=config.data_dir,
        artifacts_dir=config.artifacts_dir,
        models_dir=config.artifacts_dir / "models_placeholder",
    )
    training_config.split.validation_days = config.validation_days
    frames = prepare_training_frames(training_config)

    metadata = {
        "run_id": config.run_id,
        "transformed_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_days": config.validation_days,
        "events_prepared_rows": int(len(frames["events"])),
        "train_events_rows": int(len(frames["train_events"])),
        "validation_events_rows": int(len(frames["validation_events"])),
        "aggregated_train_rows": int(len(frames["aggregated_train"])),
        "item_snapshot_rows": int(len(frames["item_snapshot"])),
        "evaluation_frame_rows": int(len(frames["eval_frame"])),
    }
    save_json(metadata, config.artifacts_dir / "reports" / "etl_summary.json")
    return metadata


def _events_prepared_payload(events: pd.DataFrame) -> pd.DataFrame:
    payload = events.copy()
    payload["event"] = payload["event"].astype("string")
    payload["event_date"] = pd.to_datetime(payload["timestamp"], unit="ms").dt.date.astype("string")
    return payload


def _evaluation_payload(eval_frame: pd.DataFrame) -> pd.DataFrame:
    payload = eval_frame.copy()
    payload["target_items_json"] = payload["target_items"].apply(lambda x: list(map(int, x)))
    payload["purchased_items_json"] = payload["purchased_items"].apply(lambda x: list(map(int, x)))
    return payload.drop(columns=["target_items", "purchased_items"])


def _etl_run_payload(summary: dict[str, str | int]) -> pd.DataFrame:
    return pd.DataFrame([summary])


def _write_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    table_name: str,
    schema: str,
    dtype: dict[str, object],
    if_exists: str,
    chunksize: int,
) -> None:
    metadata = MetaData(schema=schema)
    table = Table(
        table_name,
        metadata,
        *(Column(column_name, column_type) for column_name, column_type in dtype.items()),
    )

    if if_exists == "replace":
        table.drop(bind=engine, checkfirst=True)
    if if_exists not in {"replace", "append"}:
        raise ValueError(f"Unsupported if_exists mode: {if_exists}")

    table.create(bind=engine, checkfirst=True)

    with engine.begin() as connection:
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            if chunk.empty:
                continue
            connection.execute(table.insert(), chunk.to_dict(orient="records"))


def load_transformed_tables(
    engine: Engine,
    artifacts_dir: Path,
    staging_schema: str,
    mart_schema: str,
) -> dict[str, int]:
    ensure_schema(engine, staging_schema)
    ensure_schema(engine, mart_schema)

    events_prepared = _events_prepared_payload(read_parquet(artifacts_dir / "data" / "events_prepared.parquet"))
    aggregated_train = read_parquet(artifacts_dir / "data" / "aggregated_train.parquet")
    item_snapshot = read_parquet(artifacts_dir / "data" / "item_snapshot.parquet")
    evaluation_frame = _evaluation_payload(read_parquet(artifacts_dir / "data" / "evaluation_frame.parquet"))
    etl_summary = _etl_run_payload(
        {
            **load_json(artifacts_dir / "reports" / "etl_summary.json"),
            "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    _write_dataframe(
        engine,
        events_prepared,
        table_name="events_prepared",
        schema=staging_schema,
        dtype={
            "timestamp": BIGINT(),
            "visitorid": INTEGER(),
            "event": TEXT(),
            "itemid": INTEGER(),
            "event_weight": FLOAT(),
            "event_date": TEXT(),
        },
        if_exists="replace",
        chunksize=50_000,
    )
    _write_dataframe(
        engine,
        aggregated_train,
        table_name="user_item_features",
        schema=mart_schema,
        dtype={
            "visitorid": INTEGER(),
            "itemid": INTEGER(),
            "event_score": FLOAT(),
            "last_timestamp": BIGINT(),
            "view_count": SMALLINT(),
            "cart_count": SMALLINT(),
            "transaction_count": SMALLINT(),
        },
        if_exists="replace",
        chunksize=50_000,
    )
    _write_dataframe(
        engine,
        item_snapshot,
        table_name="item_snapshot",
        schema=mart_schema,
        dtype={
            "itemid": INTEGER(),
            "categoryid": INTEGER(),
            "available": SMALLINT(),
        },
        if_exists="replace",
        chunksize=50_000,
    )
    _write_dataframe(
        engine,
        evaluation_frame,
        table_name="validation_targets",
        schema=mart_schema,
        dtype={
            "visitorid": INTEGER(),
            "history_events": INTEGER(),
            "target_items_json": JSON(),
            "purchased_items_json": JSON(),
        },
        if_exists="replace",
        chunksize=20_000,
    )
    _write_dataframe(
        engine,
        etl_summary,
        table_name="etl_runs",
        schema=mart_schema,
        dtype={
            "run_id": TEXT(),
            "transformed_at_utc": TEXT(),
            "validation_days": INTEGER(),
            "events_prepared_rows": BIGINT(),
            "train_events_rows": BIGINT(),
            "validation_events_rows": BIGINT(),
            "aggregated_train_rows": BIGINT(),
            "item_snapshot_rows": BIGINT(),
            "evaluation_frame_rows": BIGINT(),
            "loaded_at_utc": TEXT(),
        },
        if_exists="append",
        chunksize=1_000,
    )

    return {
        f"{staging_schema}.events_prepared": int(len(events_prepared)),
        f"{mart_schema}.user_item_features": int(len(aggregated_train)),
        f"{mart_schema}.item_snapshot": int(len(item_snapshot)),
        f"{mart_schema}.validation_targets": int(len(evaluation_frame)),
        f"{mart_schema}.etl_runs": int(len(etl_summary)),
    }


def validate_loaded_tables(engine: Engine, staging_schema: str, mart_schema: str) -> dict[str, int]:
    queries = {
        f"{staging_schema}.events_prepared": f"SELECT EXISTS (SELECT 1 FROM {staging_schema}.events_prepared LIMIT 1)",
        f"{mart_schema}.user_item_features": f"SELECT EXISTS (SELECT 1 FROM {mart_schema}.user_item_features LIMIT 1)",
        f"{mart_schema}.item_snapshot": f"SELECT EXISTS (SELECT 1 FROM {mart_schema}.item_snapshot LIMIT 1)",
        f"{mart_schema}.validation_targets": f"SELECT EXISTS (SELECT 1 FROM {mart_schema}.validation_targets LIMIT 1)",
    }
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for table_name, query in queries.items():
            counts[table_name] = int(bool(connection.execute(text(query)).scalar_one()))
    return counts


def ensure_etl_dirs(artifacts_dir: Path) -> None:
    for relative in ["data", "reports"]:
        ensure_dir(artifacts_dir / relative)
