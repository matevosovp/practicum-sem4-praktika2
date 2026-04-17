from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SplitConfig


MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000


@dataclass(slots=True)
class TemporalSplit:
    train_events: pd.DataFrame
    validation_events: pd.DataFrame
    cutoff_timestamp: int


def infer_cutoff(events: pd.DataFrame, validation_days: int) -> int:
    max_timestamp = int(events["timestamp"].max())
    return max_timestamp - validation_days * MILLISECONDS_PER_DAY


def make_time_split(events: pd.DataFrame, config: SplitConfig) -> TemporalSplit:
    cutoff = infer_cutoff(events, config.validation_days)
    train_events = events[events["timestamp"] < cutoff].copy()
    validation_events = events[events["timestamp"] >= cutoff].copy()
    return TemporalSplit(train_events=train_events, validation_events=validation_events, cutoff_timestamp=cutoff)


def build_eval_frame(split: TemporalSplit, config: SplitConfig) -> pd.DataFrame:
    train_events = split.train_events
    validation_events = split.validation_events

    history_counts = train_events.groupby("visitorid").size().rename("history_events")
    positives = validation_events[validation_events["event"].isin(config.target_events)].copy()
    if positives.empty:
        return pd.DataFrame(columns=["visitorid", "target_items", "history_events", "purchased_items"])

    target_items = positives.groupby("visitorid")["itemid"].agg(lambda s: sorted(set(map(int, s)))).rename("target_items")
    purchased_items = (
        train_events[train_events["event"] == config.purchased_event]
        .groupby("visitorid")["itemid"]
        .agg(lambda s: sorted(set(map(int, s))))
        .rename("purchased_items")
    )

    eval_frame = pd.concat([target_items, history_counts, purchased_items], axis=1).reset_index()
    eval_frame["history_events"] = eval_frame["history_events"].fillna(0).astype("int32")
    eval_frame["target_items"] = eval_frame["target_items"].apply(lambda x: x if isinstance(x, list) else [])
    eval_frame["purchased_items"] = eval_frame["purchased_items"].apply(lambda x: x if isinstance(x, list) else [])
    eval_frame = eval_frame[eval_frame["history_events"] >= config.min_history_events].copy()
    eval_frame["target_items"] = eval_frame.apply(
        lambda row: [item for item in row["target_items"] if item not in set(row["purchased_items"])],
        axis=1,
    )
    eval_frame = eval_frame[eval_frame["target_items"].map(bool)].copy()
    return eval_frame.reset_index(drop=True)
