from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_recsys.config import TrainingConfig
from ecommerce_recsys.mlflow_utils import configure_mlflow, log_artifact, log_metrics, log_params, optional_mlflow_run
from ecommerce_recsys.pipeline import train_and_serialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ecommerce recommender.")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--max-user-history", type=int, default=20)
    parser.add_argument("--max-neighbors", type=int, default=80)
    parser.add_argument("--min-pair-score", type=float, default=1.0)
    parser.add_argument("--experiment-name", type=str, default="ecommerce-recsys")
    parser.add_argument("--run-name", type=str, default="full-train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        models_dir=args.models_dir,
        top_k=args.top_k,
        max_user_history=args.max_user_history,
        max_neighbors=args.max_neighbors,
        min_pair_score=args.min_pair_score,
    )
    config.split.validation_days = args.validation_days

    configure_mlflow(experiment_name=args.experiment_name)
    with optional_mlflow_run(
        run_name=args.run_name,
        tags={"pipeline": "training", "cwd": os.getcwd()},
    ):
        log_params(
            {
                "top_k": config.top_k,
                "validation_days": config.split.validation_days,
                "max_user_history": config.max_user_history,
                "max_neighbors": config.max_neighbors,
                "min_pair_score": config.min_pair_score,
            }
        )
        bundle = train_and_serialize(config)
        for model_name, metrics in bundle["all_metrics"].items():
            log_metrics({f"{model_name}_{metric_name}": metric_value for metric_name, metric_value in metrics.items()})
        log_artifact(str(config.artifacts_dir / "reports" / "evaluation_summary.json"))
        log_artifact(str(config.models_dir / "recommender.joblib"))

    print(f"Best model: {bundle['model_name']}")
    for metric_name, metric_value in bundle["metrics"].items():
        print(f"{metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
