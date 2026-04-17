from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_RANDOM_STATE = 42


@dataclass(slots=True)
class EventWeights:
    view: float = 1.0
    addtocart: float = 4.0
    transaction: float = 6.0

    def as_dict(self) -> dict[str, float]:
        return {
            "view": self.view,
            "addtocart": self.addtocart,
            "transaction": self.transaction,
        }


@dataclass(slots=True)
class SplitConfig:
    validation_days: int = 14
    target_events: tuple[str, ...] = ("addtocart", "transaction")
    min_history_events: int = 1
    purchased_event: str = "transaction"


@dataclass(slots=True)
class TrainingConfig:
    data_dir: Path = Path(".")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    top_k: int = 10
    max_user_history: int = 20
    max_neighbors: int = 80
    min_pair_score: float = 1.0
    recent_popularity_days: int = 30
    event_weights: EventWeights = field(default_factory=EventWeights)
    split: SplitConfig = field(default_factory=SplitConfig)

