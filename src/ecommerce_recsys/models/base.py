from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Recommendation:
    item_id: int
    score: float


class BaseRecommender:
    model_name = "base"

    def recommend(
        self,
        user_id: int | None,
        k: int = 10,
        user_history: list[dict[str, float | int]] | None = None,
        exclude_items: set[int] | None = None,
    ) -> list[Recommendation]:
        raise NotImplementedError
