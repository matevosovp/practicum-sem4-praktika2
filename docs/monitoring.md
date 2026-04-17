# Monitoring

## Service metrics

The API exports Prometheus-compatible metrics on `GET /metrics`.

Tracked metrics:

- `recsys_api_requests_total{endpoint, status}`: request volume and error-rate split by endpoint
- `recsys_api_recommendation_latency_seconds`: latency histogram for recommendation generation
- `recsys_api_fallback_total`: number of requests that required fallback logic

## Recommended dashboards

- availability: ratio of `2xx` to all requests
- latency: p50 / p95 / p99 from the histogram
- fallback share: `fallback_total / recommendation_requests_total`
- traffic: requests per minute

## Model-oriented signals

The service exposes technical metrics directly. Model quality drift is tracked outside the online process via scheduled retraining and offline evaluation artifacts:

- `artifacts/reports/evaluation_summary.json`
- MLflow experiment history
- DAG execution logs in Airflow

## Alert examples

- error rate above 2% for 10 minutes
- p95 latency above 500 ms
- fallback share spikes materially above the project baseline
- scheduled retraining fails or stops updating the model artifact
