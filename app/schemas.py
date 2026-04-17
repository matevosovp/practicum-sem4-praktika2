from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryEvent(BaseModel):
    item_id: int = Field(..., ge=1)
    event: Literal["view", "addtocart", "transaction"] = "view"
    timestamp: int | None = Field(default=None, description="Unix timestamp in milliseconds")


class RecommendationRequest(BaseModel):
    user_id: int | None = Field(default=None, description="Known visitor identifier")
    k: int = Field(default=10, ge=1, le=50)
    user_history: list[HistoryEvent] | None = None
    exclude_item_ids: list[int] = Field(default_factory=list)
    filter_bought: bool = True


class RecommendationItem(BaseModel):
    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    user_id: int | None
    fallback_used: bool
    recommendations: list[RecommendationItem]
