from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .base import BaseRecommender, Recommendation


class PopularityRecommender(BaseRecommender):
    model_name = "global_popularity"

    def __init__(self, popular_items: list[tuple[int, float]]):
        self.popular_items = popular_items

    @classmethod
    def fit(cls, events: pd.DataFrame, limit: int = 1_000) -> "PopularityRecommender":
        popularity = (
            events.groupby("itemid", observed=True)["event_weight"]
            .sum()
            .sort_values(ascending=False)
            .head(limit)
        )
        popular_items = [(int(item_id), float(score)) for item_id, score in popularity.items()]
        return cls(popular_items=popular_items)

    def recommend(
        self,
        user_id: int | None,
        k: int = 10,
        user_history: list[dict[str, float | int]] | None = None,
        exclude_items: set[int] | None = None,
    ) -> list[Recommendation]:
        exclude_items = exclude_items or set()
        result: list[Recommendation] = []
        for item_id, score in self.popular_items:
            if item_id in exclude_items:
                continue
            result.append(Recommendation(item_id=item_id, score=score))
            if len(result) >= k:
                break
        return result
