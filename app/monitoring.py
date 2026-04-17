from __future__ import annotations

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "recsys_api_requests_total",
    "Total number of requests to the recommendation API",
    ["endpoint", "status"],
)
FALLBACK_COUNT = Counter(
    "recsys_api_fallback_total",
    "Number of recommendation requests that used fallback logic",
)
RECOMMENDATION_LATENCY = Histogram(
    "recsys_api_recommendation_latency_seconds",
    "Latency for generating recommendations",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)
