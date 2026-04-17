from __future__ import annotations

from collections import Counter
from pathlib import Path

from ecommerce_recsys.config import EventWeights
from ecommerce_recsys.serialization import load_joblib


class RecommendationService:
    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self.bundle = load_joblib(bundle_path)
        self.model = self.bundle["model"]
        self.model_name = self.bundle["model_name"]
        self.serving_history_frame = self.bundle["serving_history_frame"].set_index("visitorid", drop=False)
        self.purchased_items = self.bundle["purchased_items"]
        self.available_items = self.bundle.get("available_items", set())
        self.event_weights = EventWeights().as_dict()

    def _normalize_history(self, history_payload: list[dict] | None) -> list[dict[str, float | int]] | None:
        if not history_payload:
            return None
        score_counter: Counter[int] = Counter()
        last_seen: dict[int, int] = {}
        for entry in history_payload:
            item_id = int(entry["item_id"])
            event_name = entry.get("event", "view")
            score_counter[item_id] += float(self.event_weights.get(event_name, 1.0))
            if entry.get("timestamp") is not None:
                last_seen[item_id] = int(entry["timestamp"])

        normalized = [
            {
                "item_id": item_id,
                "event_score": float(score),
                "last_timestamp": int(last_seen.get(item_id, 0)),
                "view_count": 0,
                "cart_count": 0,
                "transaction_count": 0,
            }
            for item_id, score in score_counter.most_common()
        ]
        normalized.sort(key=lambda x: (x["last_timestamp"], x["event_score"]), reverse=True)
        return normalized

    def _history_from_store(self, user_id: int | None) -> list[dict[str, float | int]] | None:
        if user_id is None or user_id not in self.serving_history_frame.index:
            return None
        rows = self.serving_history_frame.loc[user_id]
        if hasattr(rows, "to_dict") and not hasattr(rows, "iterrows"):
            rows = rows.to_frame().T
        history = []
        for row in rows.itertuples(index=False):
            history.append(
                {
                    "item_id": int(row.itemid),
                    "event_score": float(row.event_score),
                    "last_timestamp": int(row.last_timestamp),
                    "view_count": int(row.view_count),
                    "cart_count": int(row.cart_count),
                    "transaction_count": int(row.transaction_count),
                }
            )
        return history

    def recommend(
        self,
        user_id: int | None,
        k: int,
        user_history: list[dict] | None = None,
        exclude_item_ids: list[int] | None = None,
        filter_bought: bool = True,
    ) -> tuple[list[dict[str, float | int]], bool]:
        profile = self._normalize_history(user_history) if user_history else self._history_from_store(user_id)
        exclude = set(exclude_item_ids or [])

        if filter_bought and user_id is not None:
            exclude.update(self.purchased_items.get(user_id, set()))

        recommendations = self.model.recommend(
            user_id=user_id,
            k=max(k * 3, k),
            user_history=profile,
            exclude_items=exclude,
        )
        filtered = []
        for rec in recommendations:
            if self.available_items and rec.item_id not in self.available_items:
                continue
            filtered.append({"item_id": rec.item_id, "score": round(rec.score, 6)})
            if len(filtered) >= k:
                break

        fallback_used = profile is None or len(filtered) < k
        if len(filtered) < k:
            for item_id, score in self.bundle["global_popularity"]:
                if item_id in exclude or any(existing["item_id"] == item_id for existing in filtered):
                    continue
                if self.available_items and item_id not in self.available_items:
                    continue
                filtered.append({"item_id": item_id, "score": round(float(score), 6)})
                if len(filtered) >= k:
                    break

        return filtered[:k], fallback_used
