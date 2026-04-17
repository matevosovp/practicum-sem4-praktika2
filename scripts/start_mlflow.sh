#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
BACKEND_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///$ROOT_DIR/artifacts/mlflow/mlflow.db}"

mkdir -p "$ROOT_DIR/artifacts/mlflow" "$ROOT_DIR/artifacts/mlruns"

if [[ -n "${S3_BUCKET_NAME:-}" ]]; then
  ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-s3://$S3_BUCKET_NAME/mlflow}"
else
  ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$ROOT_DIR/artifacts/mlruns}"
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://$MLFLOW_HOST:$MLFLOW_PORT}"

echo "MLflow backend: $BACKEND_URI"
echo "MLflow artifact root: $ARTIFACT_ROOT"
echo "MLflow tracking uri: $MLFLOW_TRACKING_URI"

mlflow server \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT" \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$ARTIFACT_ROOT"
