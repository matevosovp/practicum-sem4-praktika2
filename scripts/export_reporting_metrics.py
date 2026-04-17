from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_recsys.db import create_psycopg_connection, resolve_database_config
from ecommerce_recsys.reporting import build_artifact_registry, build_reporting_payload, load_reporting_to_db, write_reporting_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact reporting metrics and optionally persist them to Postgres.")
    parser.add_argument("--run-id", type=str, default="manual-report")
    parser.add_argument("--mart-schema", type=str, default="recsys_mart")
    parser.add_argument("--persist-db", action="store_true")
    parser.add_argument("--conn-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_reporting_payload(ROOT)
    write_reporting_payload(ROOT, payload)

    result: dict[str, object] = {"reporting_metrics": payload}
    if args.persist_db:
        db_config = resolve_database_config(conn_id=args.conn_id)
        conn = create_psycopg_connection(db_config)
        artifact_registry = build_artifact_registry(ROOT, run_id=args.run_id)
        db_result = load_reporting_to_db(
            conn=conn,
            mart_schema=args.mart_schema,
            run_id=args.run_id,
            payload=payload,
            artifact_registry=artifact_registry,
        )
        conn.close()
        result["db_load"] = db_result

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
