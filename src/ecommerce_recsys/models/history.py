from __future__ import annotations

from collections import defaultdict

from .base import BaseRecommender, Recommendation


class HistoryRecommender(BaseRecommender):
    model_name = "history_baseline"

    def __init__(self, fallback_items: list[tuple[int, float]]):
        self.fallback_items = fallback_items

    def recommend(
        self,
        user_id: int | None,
        k: int = 10,
        user_history: list[dict[str, float | int]] | None = None,
        exclude_items: set[int] | None = None,
    ) -> list[Recommendation]:
        exclude_items = exclude_items or set()
        scores = defaultdict(float)
        if user_history:
            for idx, record in enumerate(user_history):
                item_id = int(record["item_id"])
                recency_bonus = max(0.1, 1.0 - idx * 0.05)
                score = float(record["event_score"]) * recency_bonus
                scores[item_id] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result: list[Recommendation] = []

        for item_id, score in ranked:
            if item_id in exclude_items:
                continue
            result.append(Recommendation(item_id=item_id, score=float(score)))
            if len(result) >= k:
                return result

        for item_id, score in self.fallback_items:
            if item_id in exclude_items or any(rec.item_id == item_id for rec in result):
                continue
            result.append(Recommendation(item_id=item_id, score=float(score)))
            if len(result) >= k:
                break
        return result
