from __future__ import annotations

from collections import defaultdict
from math import log1p

import numpy as np
import pandas as pd

from .base import BaseRecommender, Recommendation


class WeightedCooccurrenceRecommender(BaseRecommender):
    model_name = "weighted_item2item"

    def __init__(
        self,
        item_neighbors: dict[int, list[tuple[int, float]]],
        fallback_items: list[tuple[int, float]],
    ) -> None:
        self.item_neighbors = item_neighbors
        self.fallback_items = fallback_items

    @classmethod
    def fit(
        cls,
        aggregated_events: pd.DataFrame,
        fallback_items: list[tuple[int, float]],
        max_history: int = 20,
        max_neighbors: int = 80,
        min_pair_score: float = 1.0,
    ) -> "WeightedCooccurrenceRecommender":
        neighbor_scores: dict[int, dict[int, float]] = defaultdict(dict)
        sorted_events = aggregated_events.sort_values(
            ["visitorid", "last_timestamp", "event_score"],
            ascending=[True, False, False],
        )
        sorted_events["history_rank"] = sorted_events.groupby("visitorid", observed=True).cumcount()
        sorted_events = sorted_events[sorted_events["history_rank"] < max_history].copy()
        multi_item_users = (
            sorted_events.groupby("visitorid", observed=True)["itemid"].size().loc[lambda s: s > 1].index
        )
        sorted_events = sorted_events[sorted_events["visitorid"].isin(multi_item_users)]

        for _, group in sorted_events.groupby("visitorid", sort=False, observed=True):
            items = group["itemid"].to_numpy(dtype=np.int32, copy=False)
            scores = group["event_score"].to_numpy(dtype=np.float32, copy=False)
            if len(items) < 2:
                continue

            norm = 1.0 / log1p(len(items) + 1)
            for left_idx, left_item in enumerate(items[:-1]):
                left_score = float(scores[left_idx])
                for right_idx in range(left_idx + 1, len(items)):
                    right_item = int(items[right_idx])
                    pair_score = float(left_score * float(scores[right_idx]) * norm)
                    if pair_score < min_pair_score:
                        continue
                    neighbor_scores[int(left_item)][right_item] = neighbor_scores[int(left_item)].get(right_item, 0.0) + pair_score
                    neighbor_scores[right_item][int(left_item)] = neighbor_scores[right_item].get(int(left_item), 0.0) + pair_score

        item_neighbors: dict[int, list[tuple[int, float]]] = {}
        for item_id, neighbors in neighbor_scores.items():
            ranked = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:max_neighbors]
            item_neighbors[int(item_id)] = [(int(other_id), float(score)) for other_id, score in ranked]
        return cls(item_neighbors=item_neighbors, fallback_items=fallback_items)

    def recommend(
        self,
        user_id: int | None,
        k: int = 10,
        user_history: list[dict[str, float | int]] | None = None,
        exclude_items: set[int] | None = None,
    ) -> list[Recommendation]:
        exclude_items = exclude_items or set()
        score_map: dict[int, float] = defaultdict(float)

        if user_history:
            for rank, record in enumerate(user_history):
                item_id = int(record["item_id"])
                if item_id not in self.item_neighbors:
                    continue
                base_score = float(record["event_score"]) * max(0.15, 1.0 - rank * 0.05)
                for neighbor_id, neighbor_score in self.item_neighbors[item_id]:
                    if neighbor_id in exclude_items or neighbor_id == item_id:
                        continue
                    score_map[neighbor_id] += base_score * neighbor_score

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        result: list[Recommendation] = []
        seen = set()

        for item_id, score in ranked:
            if item_id in exclude_items or item_id in seen:
                continue
            result.append(Recommendation(item_id=int(item_id), score=float(score)))
            seen.add(int(item_id))
            if len(result) >= k:
                return result

        for item_id, score in self.fallback_items:
            if item_id in exclude_items or item_id in seen:
                continue
            result.append(Recommendation(item_id=item_id, score=score))
            seen.add(item_id)
            if len(result) >= k:
                break
        return result
