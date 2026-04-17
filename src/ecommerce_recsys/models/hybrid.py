from __future__ import annotations

from collections import defaultdict

from .base import BaseRecommender, Recommendation


class HybridRecommender(BaseRecommender):
    model_name = "hybrid_history_item2item"

    def __init__(
        self,
        item_neighbors: dict[int, list[tuple[int, float]]],
        fallback_items: list[tuple[int, float]],
        history_weight: float = 1.0,
        neighbor_weight: float = 0.15,
    ) -> None:
        self.item_neighbors = item_neighbors
        self.fallback_items = fallback_items
        self.history_weight = history_weight
        self.neighbor_weight = neighbor_weight

    def recommend(
        self,
        user_id: int | None,
        k: int = 10,
        user_history: list[dict[str, float | int]] | None = None,
        exclude_items: set[int] | None = None,
    ) -> list[Recommendation]:
        exclude_items = exclude_items or set()
        scores = defaultdict(float)
        seen = set()

        if user_history:
            for rank, record in enumerate(user_history):
                item_id = int(record["item_id"])
                base_score = float(record["event_score"]) * max(0.1, 1.0 - rank * 0.05)
                if item_id not in exclude_items:
                    scores[item_id] += self.history_weight * base_score
                seen.add(item_id)
                for neighbor_id, neighbor_score in self.item_neighbors.get(item_id, []):
                    if neighbor_id in exclude_items:
                        continue
                    scores[neighbor_id] += self.neighbor_weight * base_score * float(neighbor_score)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result: list[Recommendation] = []
        used = set()
        for item_id, score in ranked:
            if item_id in exclude_items or item_id in used:
                continue
            result.append(Recommendation(item_id=int(item_id), score=float(score)))
            used.add(int(item_id))
            if len(result) >= k:
                return result

        for item_id, score in self.fallback_items:
            if item_id in exclude_items or item_id in used:
                continue
            result.append(Recommendation(item_id=item_id, score=float(score)))
            if len(result) >= k:
                break
        return result
