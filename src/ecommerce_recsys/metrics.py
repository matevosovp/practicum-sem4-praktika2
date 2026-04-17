from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def recall_at_k(actual: Iterable[int], predicted: list[int], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    predicted_k = predicted[:k]
    hits = len(actual_set.intersection(predicted_k))
    return hits / len(actual_set)


def hit_rate_at_k(actual: Iterable[int], predicted: list[int], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    return float(any(item in actual_set for item in predicted[:k]))


def ndcg_at_k(actual: Iterable[int], predicted: list[int], k: int) -> float:
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_predictions(eval_rows: list[dict], k: int) -> dict[str, float]:
    recalls = [recall_at_k(row["actual"], row["predicted"], k) for row in eval_rows]
    hit_rates = [hit_rate_at_k(row["actual"], row["predicted"], k) for row in eval_rows]
    ndcgs = [ndcg_at_k(row["actual"], row["predicted"], k) for row in eval_rows]
    coverage_items = set()
    for row in eval_rows:
        coverage_items.update(row["predicted"][:k])
    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": float(len(eval_rows)),
        "catalog_coverage": float(len(coverage_items)),
    }
