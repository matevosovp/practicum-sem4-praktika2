from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_recsys.config import ETLConfig
from ecommerce_recsys.db import create_postgres_engine, resolve_database_config, test_connection
from ecommerce_recsys.etl import ensure_etl_dirs, load_transformed_tables, run_local_transform, validate_loaded_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local transform and load transformed tables to Postgres.")
    parser.add_argument("command", choices=["transform", "load", "validate", "full"])
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--staging-schema", type=str, default="recsys_staging")
    parser.add_argument("--mart-schema", type=str, default="recsys_mart")
    parser.add_argument("--run-id", type=str, default="manual")
    parser.add_argument("--conn-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    etl_config = ETLConfig(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        validation_days=args.validation_days,
        staging_schema=args.staging_schema,
        mart_schema=args.mart_schema,
        run_id=args.run_id,
    )
    ensure_etl_dirs(etl_config.artifacts_dir)

    if args.command == "transform":
        summary = run_local_transform(etl_config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    db_config = resolve_database_config(conn_id=args.conn_id)
    engine = create_postgres_engine(db_config)
    test_connection(engine)

    if args.command == "load":
        result = load_transformed_tables(
            engine=engine,
            artifacts_dir=etl_config.artifacts_dir,
            staging_schema=etl_config.staging_schema,
            mart_schema=etl_config.mart_schema,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        result = validate_loaded_tables(engine, etl_config.staging_schema, etl_config.mart_schema)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = run_local_transform(etl_config)
    load_result = load_transformed_tables(
        engine=engine,
        artifacts_dir=etl_config.artifacts_dir,
        staging_schema=etl_config.staging_schema,
        mart_schema=etl_config.mart_schema,
    )
    validate_result = validate_loaded_tables(engine, etl_config.staging_schema, etl_config.mart_schema)
    print(
        json.dumps(
            {"transform": summary, "load": load_result, "validate": validate_result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
