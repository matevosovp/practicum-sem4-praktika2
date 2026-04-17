from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TrainingConfig
from .data import add_event_weights, load_events, load_item_snapshot, save_parquet
from .features import aggregate_user_item, build_serving_history_frame, build_user_profiles, collect_purchased_items
from .metrics import evaluate_predictions
from .models import HybridRecommender, HistoryRecommender, PopularityRecommender, WeightedCooccurrenceRecommender
from .serialization import save_joblib, save_json
from .split import build_eval_frame, make_time_split


def prepare_training_frames(config: TrainingConfig) -> dict[str, pd.DataFrame]:
    events = load_events(config.data_dir / "events.csv")
    events = add_event_weights(events, config.event_weights)
    split = make_time_split(events, config.split)
    eval_frame = build_eval_frame(split, config.split)
    aggregated_train = aggregate_user_item(split.train_events)

    item_snapshot = load_item_snapshot(
        [
            config.data_dir / "item_properties_part1.csv",
            config.data_dir / "item_properties_part2.csv",
        ]
    )

    save_parquet(split.train_events, config.artifacts_dir / "data" / "train_events.parquet")
    save_parquet(split.validation_events, config.artifacts_dir / "data" / "validation_events.parquet")
    save_parquet(aggregated_train, config.artifacts_dir / "data" / "aggregated_train.parquet")
    save_parquet(eval_frame, config.artifacts_dir / "data" / "evaluation_frame.parquet")
    save_parquet(item_snapshot, config.artifacts_dir / "data" / "item_snapshot.parquet")

    return {
        "events": events,
        "train_events": split.train_events,
        "validation_events": split.validation_events,
        "aggregated_train": aggregated_train,
        "eval_frame": eval_frame,
        "item_snapshot": item_snapshot,
        "cutoff_timestamp": pd.DataFrame({"cutoff_timestamp": [split.cutoff_timestamp]}),
    }


def fit_models(config: TrainingConfig, aggregated_train: pd.DataFrame, train_events: pd.DataFrame) -> dict[str, object]:
    popularity_model = PopularityRecommender.fit(train_events, limit=2_000)
    fallback_items = popularity_model.popular_items
    history_model = HistoryRecommender(fallback_items=fallback_items)
    cooccurrence_model = WeightedCooccurrenceRecommender.fit(
        aggregated_events=aggregated_train,
        fallback_items=fallback_items,
        max_history=config.max_user_history,
        max_neighbors=config.max_neighbors,
        min_pair_score=config.min_pair_score,
    )
    hybrid_model = HybridRecommender(
        item_neighbors=cooccurrence_model.item_neighbors,
        fallback_items=fallback_items,
    )
    return {
        popularity_model.model_name: popularity_model,
        history_model.model_name: history_model,
        cooccurrence_model.model_name: cooccurrence_model,
        hybrid_model.model_name: hybrid_model,
    }


def evaluate_model(
    model,
    eval_frame: pd.DataFrame,
    user_profiles: dict[int, list[dict[str, float | int]]],
    purchased_items: dict[int, set[int]],
    k: int,
) -> tuple[dict[str, float], list[dict]]:
    eval_rows: list[dict] = []
    for row in eval_frame.itertuples(index=False):
        user_id = int(row.visitorid)
        history = user_profiles.get(user_id)
        predictions = model.recommend(
            user_id=user_id,
            k=k,
            user_history=history,
            exclude_items=set(purchased_items.get(user_id, set())),
        )
        eval_rows.append(
            {
                "user_id": user_id,
                "actual": list(map(int, row.target_items)),
                "predicted": [rec.item_id for rec in predictions],
            }
        )
    return evaluate_predictions(eval_rows, k=k), eval_rows


def train_and_serialize(config: TrainingConfig) -> dict:
    frames = prepare_training_frames(config)
    aggregated_train = frames["aggregated_train"]
    train_events = frames["train_events"]
    eval_frame = frames["eval_frame"]
    item_snapshot = frames["item_snapshot"]

    models = fit_models(config, aggregated_train=aggregated_train, train_events=train_events)
    eval_user_ids = set(eval_frame["visitorid"].astype("int32").tolist())
    eval_profiles = build_user_profiles(
        aggregated_train,
        max_history=config.max_user_history,
        allowed_users=eval_user_ids,
    )
    serving_history_frame = build_serving_history_frame(aggregated_train, max_history=config.max_user_history, min_items=2)
    purchased_items = collect_purchased_items(train_events)
    if "available" in item_snapshot.columns:
        available_items = set(
            item_snapshot.loc[item_snapshot["available"].fillna(1).astype("int8") == 1, "itemid"].astype("int32").tolist()
        )
    else:
        available_items = set(item_snapshot["itemid"].astype("int32").tolist()) if not item_snapshot.empty else set()

    metrics_by_model: dict[str, dict[str, float]] = {}
    best_model_name = ""
    best_model_score = -1.0

    for model_name, model in models.items():
        metrics, eval_rows = evaluate_model(
            model=model,
            eval_frame=eval_frame,
            user_profiles=eval_profiles,
            purchased_items=purchased_items,
            k=config.top_k,
        )
        metrics_by_model[model_name] = metrics
        if metrics["recall_at_k"] > best_model_score:
            best_model_name = model_name
            best_model_score = metrics["recall_at_k"]
            save_json(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "sample_predictions": eval_rows[:20],
                },
                config.artifacts_dir / "reports" / "best_model_preview.json",
            )

    best_model = models[best_model_name]
    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "top_k": config.top_k,
        "metrics": metrics_by_model[best_model_name],
        "all_metrics": metrics_by_model,
        "serving_history_frame": serving_history_frame,
        "purchased_items": purchased_items,
        "global_popularity": models["global_popularity"].popular_items,
        "available_items": available_items,
        "training_config": {
            "top_k": config.top_k,
            "max_user_history": config.max_user_history,
            "max_neighbors": config.max_neighbors,
            "min_pair_score": config.min_pair_score,
            "validation_days": config.split.validation_days,
        },
    }

    save_joblib(bundle, config.models_dir / "recommender.joblib")
    save_parquet(serving_history_frame, config.models_dir / "serving_history.parquet")
    save_json(
        {
            "best_model_name": best_model_name,
            "all_metrics": metrics_by_model,
            "cutoff_timestamp": int(frames["cutoff_timestamp"]["cutoff_timestamp"].iloc[0]),
            "top_k": config.top_k,
        },
        config.artifacts_dir / "reports" / "evaluation_summary.json",
    )
    return bundle
