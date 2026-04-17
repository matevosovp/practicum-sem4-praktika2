from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from .config import DatabaseConfig

DEFAULT_AIRFLOW_CONN_ID = "recsys_postgres"


def load_env_file(env_path: Path | None = None) -> None:
    load_dotenv(dotenv_path=env_path or Path(".env"), override=False)


def database_config_from_env(prefix: str = "DB_DESTINATION") -> DatabaseConfig:
    load_env_file()
    return DatabaseConfig(
        host=os.environ[f"{prefix}_HOST"],
        port=int(os.environ[f"{prefix}_PORT"]),
        database=os.environ[f"{prefix}_NAME"],
        user=os.environ[f"{prefix}_USER"],
        password=os.environ[f"{prefix}_PASSWORD"],
        sslmode=os.getenv(f"{prefix}_SSLMODE", "require"),
    )


def database_config_from_airflow_connection(conn_id: str = DEFAULT_AIRFLOW_CONN_ID) -> DatabaseConfig:
    try:
        from airflow.hooks.base import BaseHook
    except ImportError as exc:
        raise RuntimeError(
            "Airflow is not installed, so the project cannot resolve an Airflow connection."
        ) from exc

    connection = BaseHook.get_connection(conn_id)
    if not connection.host:
        raise ValueError(f"Airflow connection '{conn_id}' must define a host.")
    if not connection.schema:
        raise ValueError(f"Airflow connection '{conn_id}' must define a schema/database name.")
    if not connection.login:
        raise ValueError(f"Airflow connection '{conn_id}' must define a login/user.")

    extras = connection.extra_dejson or {}
    return DatabaseConfig(
        host=connection.host,
        port=connection.port or 5432,
        database=connection.schema,
        user=connection.login,
        password=connection.password or "",
        sslmode=str(extras.get("sslmode", "require")),
    )


def resolve_database_config(conn_id: str | None = None, prefix: str = "DB_DESTINATION") -> DatabaseConfig:
    resolved_conn_id = conn_id or os.getenv("RECSYS_AIRFLOW_CONN_ID")
    if resolved_conn_id:
        return database_config_from_airflow_connection(resolved_conn_id)
    return database_config_from_env(prefix=prefix)


def build_sqlalchemy_url(config: DatabaseConfig) -> str:
    return (
        f"postgresql+psycopg2://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
        f"?sslmode={config.sslmode}&connect_timeout=10"
    )


def create_postgres_engine(config: DatabaseConfig) -> Engine:
    return create_engine(
        build_sqlalchemy_url(config),
        poolclass=NullPool,
        connect_args={"connect_timeout": 10},
    )


def create_psycopg_connection(config: DatabaseConfig):
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        sslmode=config.sslmode,
        connect_timeout=10,
    )


def ensure_schema(engine: Engine, schema_name: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))


def test_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
