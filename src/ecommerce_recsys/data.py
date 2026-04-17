from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import EventWeights
from .constants import EVENT_USECOLS, ITEM_PROPERTY_FILTER, PROPERTY_COLUMNS


EVENT_DTYPES = {
    "timestamp": "int64",
    "visitorid": "int32",
    "event": "string",
    "itemid": "int32",
}

PROPERTY_DTYPES = {
    "timestamp": "int64",
    "itemid": "int32",
    "property": "string",
    "value": "string",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_events(events_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    usecols = usecols or EVENT_USECOLS
    events = pd.read_csv(events_path, usecols=usecols, dtype=EVENT_DTYPES)
    events["event"] = events["event"].astype("category")
    return events


def add_event_weights(events: pd.DataFrame, weights: EventWeights) -> pd.DataFrame:
    weighted = events.copy()
    weighted["event_weight"] = weighted["event"].map(weights.as_dict()).astype("float32")
    return weighted


def load_category_tree(path: Path) -> pd.DataFrame:
    tree = pd.read_csv(path)
    tree["categoryid"] = pd.to_numeric(tree["categoryid"], downcast="integer")
    tree["parentid"] = pd.to_numeric(tree["parentid"], downcast="integer")
    return tree


def iter_property_chunks(path: Path, chunksize: int = 1_000_000) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=PROPERTY_COLUMNS,
        dtype=PROPERTY_DTYPES,
    )


def load_item_snapshot(
    property_paths: list[Path],
    property_filter: set[str] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    property_filter = property_filter or ITEM_PROPERTY_FILTER
    latest_frames: list[pd.DataFrame] = []

    for path in property_paths:
        for chunk in iter_property_chunks(path, chunksize=chunksize):
            filtered = chunk[chunk["property"].isin(property_filter)].copy()
            if filtered.empty:
                continue
            filtered.sort_values("timestamp", inplace=True)
            deduped = filtered.drop_duplicates(subset=["itemid", "property"], keep="last")
            latest_frames.append(deduped)

    if not latest_frames:
        return pd.DataFrame(columns=["itemid", "categoryid", "available"])

    merged = pd.concat(latest_frames, ignore_index=True)
    merged.sort_values("timestamp", inplace=True)
    merged = merged.drop_duplicates(subset=["itemid", "property"], keep="last")
    snapshot = merged.pivot(index="itemid", columns="property", values="value").reset_index()
    snapshot.columns.name = None

    if "categoryid" in snapshot.columns:
        snapshot["categoryid"] = pd.to_numeric(snapshot["categoryid"], errors="coerce", downcast="integer")
    if "available" in snapshot.columns:
        snapshot["available"] = pd.to_numeric(snapshot["available"], errors="coerce", downcast="integer").fillna(1)

    return snapshot


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
