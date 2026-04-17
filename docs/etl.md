# ETL To Postgres

ETL pipeline uses local CSV files as source, performs transformation on the local machine, stores intermediate results in parquet, and loads only transformed tables to Postgres.

## Target schemas

- `recsys_staging`
- `recsys_mart`

## Loaded tables

### `recsys_staging.events_prepared`

Normalized event log with compact columns:

- `timestamp`
- `visitorid`
- `event`
- `itemid`
- `event_weight`
- `event_date`

### `recsys_mart.user_item_features`

Aggregated train-window features per `user-item`:

- `visitorid`
- `itemid`
- `event_score`
- `last_timestamp`
- `view_count`
- `cart_count`
- `transaction_count`

### `recsys_mart.item_snapshot`

Latest available item attributes:

- `itemid`
- `categoryid`
- `available`

### `recsys_mart.validation_targets`

Validation users for offline evaluation:

- `visitorid`
- `history_events`
- `target_items_json`
- `purchased_items_json`

### `recsys_mart.etl_runs`

Technical run metadata:

- `run_id`
- `transformed_at_utc`
- `loaded_at_utc`
- row counts of transformed datasets

### `recsys_mart.reporting_runs`

Summary of final training/reporting runs:

- `run_id`
- `created_at_utc`
- `best_model_name`
- `top_k`
- `cutoff_timestamp`
- compact JSON summaries for ETL, EDA, training and evaluation datasets

### `recsys_mart.model_metrics_history`

Per-model offline metrics saved after retraining:

- `run_id`
- `model_name`
- `recall_at_k`
- `hit_rate_at_k`
- `ndcg_at_k`
- `evaluated_users`
- `catalog_coverage`

### `recsys_mart.artifact_registry`

Registry of key saved artifacts:

- `run_id`
- `artifact_name`
- `artifact_type`
- `artifact_path`
- `file_size_bytes`
- `sha256`
- `created_at_utc`
- `registered_at_utc`

## Runtime flow

1. `scripts/run_etl.py transform`
2. `scripts/run_etl.py load`
3. `scripts/run_etl.py validate`

Airflow DAG:

- `ecommerce_recommender_etl_to_postgres`
