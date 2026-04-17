from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from ecommerce_recsys.constants import ITEM_PROPERTY_FILTER
from ecommerce_recsys.data import add_event_weights, load_category_tree, load_events, load_item_snapshot
from ecommerce_recsys.config import EventWeights
from ecommerce_recsys.serialization import save_json


def main() -> None:
    events = add_event_weights(load_events(ROOT / "events.csv"), EventWeights())
    categories = load_category_tree(ROOT / "category_tree.csv")
    item_snapshot = load_item_snapshot(
        [ROOT / "item_properties_part1.csv", ROOT / "item_properties_part2.csv"],
        property_filter=ITEM_PROPERTY_FILTER,
    )

    coverage = item_snapshot["itemid"].isin(events["itemid"].unique()).mean()
    daily = (
        events.assign(date=pd.to_datetime(events["timestamp"], unit="ms").dt.date)
        .pivot_table(index="date", columns="event", values="itemid", aggfunc="count", fill_value=0, observed=False)
        .reset_index()
    )
    summary = {
        "events_rows": int(len(events)),
        "users": int(events["visitorid"].nunique()),
        "items": int(events["itemid"].nunique()),
        "categories_rows": int(len(categories)),
        "item_snapshot_rows": int(len(item_snapshot)),
        "item_snapshot_coverage_in_events": float(round(coverage, 4)),
        "event_share": {
            event_name: float(round(count / len(events), 4))
            for event_name, count in events["event"].value_counts().to_dict().items()
        },
        "daily_event_means": {
            column: float(round(daily[column].mean(), 2))
            for column in daily.columns
            if column != "date"
        },
    }
    save_json(summary, ROOT / "artifacts" / "reports" / "eda_runtime_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
