from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_recsys.config import TrainingConfig
from ecommerce_recsys.features import build_user_profiles, collect_purchased_items
from ecommerce_recsys.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k
from ecommerce_recsys.pipeline import fit_models
from ecommerce_recsys.serialization import load_joblib, save_json


def _to_iso(timestamp_ms: int) -> str:
    return pd.to_datetime(timestamp_ms, unit="ms").isoformat()


def _frame_memory_mb(frame: pd.DataFrame) -> float:
    return float(round(frame.memory_usage(deep=True).sum() / 1024 / 1024, 2))


def _share(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(round(numerator / denominator, 4))


def _make_bucket_table(series: pd.Series, bins: list[int], labels: list[str], name: str) -> list[dict[str, float | int | str]]:
    bucketed = pd.cut(series, bins=bins, right=False, labels=labels)
    counts = bucketed.value_counts().sort_index()
    total = int(len(series))
    rows: list[dict[str, float | int | str]] = []
    for label, count in counts.items():
        rows.append(
            {
                name: str(label),
                "count": int(count),
                "share": float(round(int(count) / total, 4)) if total else 0.0,
            }
        )
    return rows


def build_eda_report() -> dict[str, object]:
    summary = json.loads((ROOT / "eda_summary.json").read_text(encoding="utf-8"))
    etl_summary = json.loads((ROOT / "artifacts" / "reports" / "etl_summary.json").read_text(encoding="utf-8"))

    events = pd.read_parquet(ROOT / "artifacts" / "data" / "events_prepared.parquet")
    aggregated_train = pd.read_parquet(ROOT / "artifacts" / "data" / "aggregated_train.parquet")
    item_snapshot = pd.read_parquet(ROOT / "artifacts" / "data" / "item_snapshot.parquet")

    monthly_events = (
        events.assign(month=pd.to_datetime(events["timestamp"], unit="ms").dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    monthly_events["month"] = monthly_events["month"].dt.strftime("%Y-%m")

    user_activity = events.groupby("visitorid", observed=True).size()
    item_activity = events.groupby("itemid", observed=True).size()

    item_activity_sorted = item_activity.sort_values(ascending=False).reset_index(drop=True)
    total_item_events = int(item_activity_sorted.sum())
    long_tail = []
    for top_n in [10, 100, 1_000, 5_000, 10_000]:
        if top_n > len(item_activity_sorted):
            continue
        share_pct = item_activity_sorted.iloc[:top_n].sum() / total_item_events * 100
        long_tail.append({"top_n": int(top_n), "event_share_pct": float(round(share_pct, 2))})

    event_items = set(events["itemid"].astype("int32").tolist())
    snapshot_items = set(item_snapshot["itemid"].astype("int32").tolist())
    top_item_coverage = []
    item_rank = item_activity.sort_values(ascending=False)
    for top_n in [100, 1_000, 5_000, 10_000]:
        if top_n > len(item_rank):
            continue
        top_items = set(item_rank.head(top_n).index.astype("int32").tolist())
        top_item_coverage.append(
            {
                "top_n": int(top_n),
                "snapshot_item_coverage": _share(len(top_items & snapshot_items), len(top_items)),
            }
        )

    event_row_coverage = events["itemid"].isin(snapshot_items)
    cart_row_coverage = events.loc[events["event"] == "addtocart", "itemid"].isin(snapshot_items)
    transaction_row_coverage = events.loc[events["event"] == "transaction", "itemid"].isin(snapshot_items)

    memory_footprint = [
        {
            "frame": "events_prepared",
            "rows": int(len(events)),
            "memory_mb": _frame_memory_mb(events),
        },
        {
            "frame": "aggregated_train",
            "rows": int(len(aggregated_train)),
            "memory_mb": _frame_memory_mb(aggregated_train),
        },
        {
            "frame": "item_snapshot",
            "rows": int(len(item_snapshot)),
            "memory_mb": _frame_memory_mb(item_snapshot),
        },
    ]

    profile_sizes = aggregated_train.groupby("visitorid", observed=True)["itemid"].size()

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "etl_summary": etl_summary,
        "dataset_overview": [
            {"table": "events.csv", "rows": int(summary["tables"]["events_rows"]), "comment": "сырой событийный лог"},
            {"table": "category_tree.csv", "rows": int(summary["tables"]["category_tree_rows"]), "comment": "иерархия категорий"},
            {
                "table": "item_properties_part1+part2.csv",
                "rows": int(summary["tables"]["item_properties_rows"]),
                "comment": "исторические свойства товаров",
            },
        ],
        "time_coverage": {
            "events_min": summary["time_coverage"]["events_min"],
            "events_max": summary["time_coverage"]["events_max"],
            "properties_min": summary["time_coverage"]["properties_min"],
            "properties_max": summary["time_coverage"]["properties_max"],
            "train_end": _to_iso(int(pd.read_parquet(ROOT / "artifacts" / "data" / "train_events.parquet", columns=["timestamp"])["timestamp"].max())),
            "validation_start": _to_iso(
                int(pd.read_parquet(ROOT / "artifacts" / "data" / "validation_events.parquet", columns=["timestamp"])["timestamp"].min())
            ),
        },
        "monthly_events": monthly_events.to_dict(orient="records"),
        "event_share_pct": [
            {"event": event_name, "share_pct": float(round(share * 100, 2))}
            for event_name, share in summary["event_shares_pct"].items()
        ],
        "funnel_user_item_pairs": [
            {"stage": "all_pairs", "pairs": int(summary["conversion"]["pairs_total"])},
            {"stage": "pairs_with_view", "pairs": int(summary["conversion"]["pairs_with_view"])},
            {"stage": "pairs_with_addtocart", "pairs": int(summary["conversion"]["pairs_with_cart"])},
            {"stage": "pairs_with_transaction", "pairs": int(summary["conversion"]["pairs_with_transaction"])},
        ],
        "funnel_rates_pct": [
            {"transition": "view_to_addtocart", "rate_pct": float(summary["conversion"]["cart_after_view_rate_pct"])},
            {"transition": "view_to_transaction", "rate_pct": float(summary["conversion"]["transaction_after_view_rate_pct"])},
            {"transition": "addtocart_to_transaction", "rate_pct": float(summary["conversion"]["transaction_after_cart_rate_pct"])},
        ],
        "user_activity_buckets": _make_bucket_table(
            user_activity,
            bins=[1, 2, 5, 10, 20, 50, 100, 10**9],
            labels=["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"],
            name="activity_bucket",
        ),
        "item_activity_buckets": _make_bucket_table(
            item_activity,
            bins=[1, 2, 5, 10, 20, 50, 100, 10**9],
            labels=["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"],
            name="activity_bucket",
        ),
        "long_tail_curve": long_tail,
        "repeat_interest": [
            {"metric": "repeat_views_share_pct", "value_pct": float(summary["conversion"]["repeat_views_share_pct"])},
            {"metric": "repeat_carts_share_pct", "value_pct": float(summary["conversion"]["repeat_carts_share_pct"])},
            {
                "metric": "aggregated_pairs_with_any_cart_pct",
                "value_pct": float(round((aggregated_train["cart_count"] >= 1).mean() * 100, 2)),
            },
            {
                "metric": "aggregated_pairs_with_any_transaction_pct",
                "value_pct": float(round((aggregated_train["transaction_count"] >= 1).mean() * 100, 2)),
            },
        ],
        "item_properties_coverage": {
            "event_item_coverage_pct": float(round(len(event_items & snapshot_items) / len(event_items) * 100, 2)),
            "event_row_coverage_pct": float(round(event_row_coverage.mean() * 100, 2)),
            "addtocart_row_coverage_pct": float(round(cart_row_coverage.mean() * 100, 2)),
            "transaction_row_coverage_pct": float(round(transaction_row_coverage.mean() * 100, 2)),
            "top_item_snapshot_coverage": top_item_coverage,
            "available_zero_share_pct": float(round((item_snapshot["available"].fillna(1).astype("int8") == 0).mean() * 100, 2)),
            "items_only_in_events": int(summary["item_properties"]["items_only_in_events"]),
            "items_only_in_properties": int(summary["item_properties"]["items_only_in_properties"]),
        },
        "memory_footprint": memory_footprint,
        "profile_size_quantiles": {
            "p50_unique_items": float(profile_sizes.quantile(0.50)),
            "p75_unique_items": float(profile_sizes.quantile(0.75)),
            "p90_unique_items": float(profile_sizes.quantile(0.90)),
            "p95_unique_items": float(profile_sizes.quantile(0.95)),
            "p99_unique_items": float(profile_sizes.quantile(0.99)),
        },
        "dtype_notes": {
            "events_prepared": {column: str(dtype) for column, dtype in events.dtypes.items()},
            "aggregated_train": {column: str(dtype) for column, dtype in aggregated_train.dtypes.items()},
            "item_snapshot": {column: str(dtype) for column, dtype in item_snapshot.dtypes.items()},
        },
    }


def build_modeling_report() -> dict[str, object]:
    bundle = load_joblib(ROOT / "models" / "recommender.joblib")
    evaluation_summary = json.loads((ROOT / "artifacts" / "reports" / "evaluation_summary.json").read_text(encoding="utf-8"))

    aggregated_train = pd.read_parquet(ROOT / "artifacts" / "data" / "aggregated_train.parquet")
    train_events = pd.read_parquet(ROOT / "artifacts" / "data" / "train_events.parquet")
    validation_events = pd.read_parquet(ROOT / "artifacts" / "data" / "validation_events.parquet")
    eval_frame = pd.read_parquet(ROOT / "artifacts" / "data" / "evaluation_frame.parquet")

    config = TrainingConfig(
        data_dir=ROOT,
        artifacts_dir=ROOT / "artifacts",
        models_dir=ROOT / "models",
        top_k=int(bundle["top_k"]),
    )
    config.split.validation_days = int(bundle["training_config"]["validation_days"])
    config.max_user_history = int(bundle["training_config"]["max_user_history"])
    config.max_neighbors = int(bundle["training_config"]["max_neighbors"])
    config.min_pair_score = float(bundle["training_config"]["min_pair_score"])

    models = fit_models(config, aggregated_train=aggregated_train, train_events=train_events)
    eval_user_ids = set(eval_frame["visitorid"].astype("int32").tolist())
    user_profiles = build_user_profiles(aggregated_train, max_history=config.max_user_history, allowed_users=eval_user_ids)
    purchased_items = collect_purchased_items(train_events)

    prediction_rows: list[dict[str, object]] = []
    for row in eval_frame.itertuples(index=False):
        user_id = int(row.visitorid)
        actual = list(map(int, row.target_items))
        history_items = [int(item["item_id"]) for item in user_profiles.get(user_id, [])]
        exclude = set(purchased_items.get(user_id, set()))
        for model_name, model in models.items():
            predicted = [
                rec.item_id
                for rec in model.recommend(
                    user_id=user_id,
                    k=config.top_k,
                    user_history=user_profiles.get(user_id),
                    exclude_items=exclude,
                )
            ]
            prediction_rows.append(
                {
                    "user_id": user_id,
                    "model_name": model_name,
                    "history_events": int(row.history_events),
                    "history_items": history_items,
                    "actual": actual,
                    "predicted": predicted,
                    "recall_at_k": float(recall_at_k(actual, predicted, config.top_k)),
                    "hit_rate_at_k": float(hit_rate_at_k(actual, predicted, config.top_k)),
                    "ndcg_at_k": float(ndcg_at_k(actual, predicted, config.top_k)),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions["history_bucket"] = pd.cut(
        predictions["history_events"],
        bins=[1, 2, 5, 10, 20, 10**9],
        right=False,
        labels=["1", "2-4", "5-9", "10-19", "20+"],
    )

    metrics_table = []
    catalog_size = int(pd.read_parquet(ROOT / "artifacts" / "data" / "events_prepared.parquet", columns=["itemid"])["itemid"].nunique())
    for model_name, metrics in evaluation_summary["all_metrics"].items():
        coverage = int(metrics["catalog_coverage"])
        metrics_table.append(
            {
                "model_name": model_name,
                "recall_at_k": float(metrics["recall_at_k"]),
                "hit_rate_at_k": float(metrics["hit_rate_at_k"]),
                "ndcg_at_k": float(metrics["ndcg_at_k"]),
                "catalog_coverage": coverage,
                "catalog_coverage_pct": float(round(coverage / catalog_size * 100, 3)),
            }
        )

    history_bucket_metrics = (
        predictions.groupby(["model_name", "history_bucket"], observed=True)
        .agg(
            users=("user_id", "nunique"),
            recall_at_k=("recall_at_k", "mean"),
            hit_rate_at_k=("hit_rate_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
        )
        .reset_index()
    )
    history_bucket_metrics["recall_at_k"] = history_bucket_metrics["recall_at_k"].round(4)
    history_bucket_metrics["hit_rate_at_k"] = history_bucket_metrics["hit_rate_at_k"].round(4)
    history_bucket_metrics["ndcg_at_k"] = history_bucket_metrics["ndcg_at_k"].round(4)

    unique_history_items = aggregated_train.groupby("visitorid", observed=True)["itemid"].size()
    eval_users = eval_frame[["visitorid", "history_events"]].copy()
    eval_users["unique_history_items"] = eval_users["visitorid"].map(unique_history_items).fillna(0).astype("int32")
    eval_users["history_target_overlap"] = eval_users["visitorid"].map(
        predictions[predictions["model_name"] == "history_baseline"]
        .set_index("user_id")
        .apply(lambda row: _share(len(set(row["history_items"]) & set(row["actual"])), len(set(row["actual"]))), axis=1)
        .to_dict()
    )

    profile_counts = aggregated_train.groupby("visitorid", observed=True)["itemid"].size()

    history_best = predictions[predictions["model_name"] == "history_baseline"].copy()
    weighted = predictions[predictions["model_name"] == "weighted_item2item"].copy()
    hybrid = predictions[predictions["model_name"] == "hybrid_history_item2item"].copy()

    joined = (
        history_best.merge(
            weighted[["user_id", "predicted", "recall_at_k"]].rename(
                columns={"predicted": "weighted_item2item", "recall_at_k": "weighted_recall_at_k"}
            ),
            on="user_id",
            how="left",
        )
        .merge(
            hybrid[["user_id", "predicted", "recall_at_k"]].rename(
                columns={"predicted": "hybrid_history_item2item", "recall_at_k": "hybrid_recall_at_k"}
            ),
            on="user_id",
            how="left",
        )
        .rename(columns={"predicted": "history_baseline"})
    )

    sample_cases: list[dict[str, object]] = []

    def _append_case(label: str, frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            return
        row = frame.sort_values(sort_columns, ascending=ascending).iloc[0]
        sample_cases.append(
            {
                "case_label": label,
                "user_id": int(row["user_id"]),
                "history_events": int(row["history_events"]),
                "history_items": list(map(int, row["history_items"][:10])),
                "actual": list(map(int, row["actual"])),
                "history_baseline": list(map(int, row["history_baseline"][:10])),
                "weighted_item2item": list(map(int, row["weighted_item2item"][:10])),
                "hybrid_history_item2item": list(map(int, row["hybrid_history_item2item"][:10])),
                "history_recall_at_k": float(row["recall_at_k"]),
                "weighted_recall_at_k": float(row["weighted_recall_at_k"]),
                "hybrid_recall_at_k": float(row["hybrid_recall_at_k"]),
            }
        )

    _append_case(
        "repeat_short_history_hit",
        joined[(joined["history_events"] == 1) & (joined["recall_at_k"] > 0)],
        ["recall_at_k", "ndcg_at_k"],
        [False, False],
    )
    _append_case(
        "repeat_medium_history_hit",
        joined[(joined["history_events"].between(5, 9)) & (joined["recall_at_k"] > 0)],
        ["recall_at_k", "ndcg_at_k"],
        [False, False],
    )
    _append_case(
        "item2item_recovers_new_item",
        joined[(joined["recall_at_k"] == 0) & (joined["weighted_recall_at_k"] > 0)],
        ["weighted_recall_at_k", "history_events"],
        [False, True],
    )
    _append_case(
        "history_beats_hybrid_on_repeat_interest",
        joined[(joined["recall_at_k"] > 0) & (joined["hybrid_recall_at_k"] == 0)],
        ["recall_at_k", "history_events"],
        [False, True],
    )
    _append_case(
        "difficult_multi_target_miss",
        joined[(joined["recall_at_k"] == 0) & (joined["actual"].map(len) >= 3)],
        ["history_events"],
        [True],
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": int(config.top_k),
            "validation_days": int(config.split.validation_days),
            "max_user_history": int(config.max_user_history),
            "max_neighbors": int(config.max_neighbors),
            "min_pair_score": float(config.min_pair_score),
            "target_events": list(config.split.target_events),
            "purchased_event": config.split.purchased_event,
        },
        "split_summary": {
            "train_rows": int(len(train_events)),
            "validation_rows": int(len(validation_events)),
            "cutoff_timestamp": int(evaluation_summary["cutoff_timestamp"]),
            "train_start": _to_iso(int(train_events["timestamp"].min())),
            "train_end": _to_iso(int(train_events["timestamp"].max())),
            "validation_start": _to_iso(int(validation_events["timestamp"].min())),
            "validation_end": _to_iso(int(validation_events["timestamp"].max())),
            "cutoff_datetime": _to_iso(int(evaluation_summary["cutoff_timestamp"])),
        },
        "evaluation_dataset": {
            "users": int(len(eval_frame)),
            "avg_history_events": float(round(eval_frame["history_events"].mean(), 2)),
            "median_history_events": float(eval_frame["history_events"].median()),
            "avg_target_items": float(round(eval_frame["target_items"].map(len).mean(), 2)),
            "share_with_repeat_target_from_history": float(round((eval_users["history_target_overlap"] > 0).mean(), 4)),
            "share_with_full_target_in_history": float(round((eval_users["history_target_overlap"] == 1).mean(), 4)),
        },
        "history_bucket_distribution": _make_bucket_table(
            eval_frame["history_events"],
            bins=[1, 2, 5, 10, 20, 10**9],
            labels=["1", "2-4", "5-9", "10-19", "20+"],
            name="history_bucket",
        ),
        "model_definitions": [
            {
                "model_name": "global_popularity",
                "family": "non-personalized baseline",
                "why_included": "нижняя граница качества и обязательный cold-start fallback",
            },
            {
                "model_name": "history_baseline",
                "family": "repeat-interest baseline",
                "why_included": "проверка гипотезы о силе короткой персональной истории в разреженном e-commerce",
            },
            {
                "model_name": "weighted_item2item",
                "family": "co-occurrence retrieval",
                "why_included": "поиск новых, но похожих товаров без dense user-item матрицы",
            },
            {
                "model_name": "hybrid_history_item2item",
                "family": "history + retrieval",
                "why_included": "компромисс между повторным интересом и каталоговым расширением",
            },
        ],
        "metrics_table": metrics_table,
        "history_bucket_metrics": history_bucket_metrics.to_dict(orient="records"),
        "serving_profile_eligibility": {
            "train_users_total": int(len(profile_counts)),
            "train_users_with_2plus_items": int((profile_counts >= 2).sum()),
            "train_users_with_2plus_items_share": float(round((profile_counts >= 2).mean(), 4)),
            "eval_users_with_2plus_items": int((eval_users["unique_history_items"] >= 2).sum()),
            "eval_users_with_2plus_items_share": float(round((eval_users["unique_history_items"] >= 2).mean(), 4)),
        },
        "sample_cases": sample_cases,
    }


def main() -> None:
    eda_report = build_eda_report()
    modeling_report = build_modeling_report()
    save_json(eda_report, ROOT / "artifacts" / "reports" / "eda_notebook_report.json")
    save_json(modeling_report, ROOT / "artifacts" / "reports" / "modeling_notebook_report.json")
    print(
        json.dumps(
            {
                "eda_notebook_report": "artifacts/reports/eda_notebook_report.json",
                "modeling_notebook_report": "artifacts/reports/modeling_notebook_report.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
