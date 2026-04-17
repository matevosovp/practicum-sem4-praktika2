from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def aggregate_user_item(events: pd.DataFrame) -> pd.DataFrame:
    base = (
        events.groupby(["visitorid", "itemid"], observed=True)
        .agg(
            event_score=("event_weight", "sum"),
            last_timestamp=("timestamp", "max"),
        )
    )
    event_counts = []
    for event_name, column_name in [
        ("view", "view_count"),
        ("addtocart", "cart_count"),
        ("transaction", "transaction_count"),
    ]:
        counts = events.loc[events["event"] == event_name].groupby(["visitorid", "itemid"], observed=True).size().rename(column_name)
        event_counts.append(counts)

    grouped = pd.concat([base, *event_counts], axis=1).fillna(0).reset_index()
    grouped["event_score"] = grouped["event_score"].astype("float32")
    for column_name in ["view_count", "cart_count", "transaction_count"]:
        grouped[column_name] = grouped[column_name].astype("int16")
    return grouped


def build_user_profiles(
    aggregated: pd.DataFrame,
    max_history: int = 20,
    min_items: int = 1,
    allowed_users: set[int] | None = None,
) -> dict[int, list[dict[str, float | int]]]:
    profiles: dict[int, list[dict[str, float | int]]] = {}
    working = aggregated
    if allowed_users is not None:
        working = working[working["visitorid"].isin(allowed_users)].copy()
    if min_items > 1 and not working.empty:
        counts = working.groupby("visitorid", observed=True)["itemid"].size()
        eligible_users = set(map(int, counts[counts >= min_items].index.tolist()))
        working = working[working["visitorid"].isin(eligible_users)].copy()
    sorted_agg = working.sort_values(["visitorid", "last_timestamp", "event_score"], ascending=[True, False, False])
    for visitor_id, group in sorted_agg.groupby("visitorid", sort=False):
        top_group = group.head(max_history)
        profiles[int(visitor_id)] = [
            {
                "item_id": int(row.itemid),
                "event_score": float(row.event_score),
                "last_timestamp": int(row.last_timestamp),
                "view_count": int(row.view_count),
                "cart_count": int(row.cart_count),
                "transaction_count": int(row.transaction_count),
            }
            for row in top_group.itertuples(index=False)
        ]
    return profiles


def build_serving_history_frame(
    aggregated: pd.DataFrame,
    max_history: int = 20,
    min_items: int = 2,
) -> pd.DataFrame:
    working = aggregated.copy()
    counts = working.groupby("visitorid", observed=True)["itemid"].size()
    eligible_users = counts[counts >= min_items].index
    working = working[working["visitorid"].isin(eligible_users)].copy()
    working.sort_values(["visitorid", "last_timestamp", "event_score"], ascending=[True, False, False], inplace=True)
    working["history_rank"] = working.groupby("visitorid", observed=True).cumcount()
    return working[working["history_rank"] < max_history].copy()


def collect_purchased_items(events: pd.DataFrame) -> dict[int, set[int]]:
    purchased = events[events["event"] == "transaction"]
    if purchased.empty:
        return {}
    grouped = purchased.groupby("visitorid")["itemid"].agg(lambda s: set(map(int, s)))
    return {int(user_id): items for user_id, items in grouped.items()}


def flatten_history_items(history: Iterable[dict[str, float | int]]) -> list[int]:
    return [int(entry["item_id"]) for entry in history]
