"""Pydantic schemas for Strategy versioning, Calibration, and Strategy Notes."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pilot.db.models import StrategyNoteStatus


# ============================================================================
# Strategy
# ============================================================================
class StrategyBase(BaseModel):
    goal_id: UUID
    version: int = Field(ge=1)
    parent_version_id: UUID | None = None
    retired_at: datetime | None = None


class StrategyCreate(StrategyBase):
    pass


class StrategyRead(StrategyBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Calibration
# ============================================================================
class CalibrationBase(BaseModel):
    strategy_id: UUID
    window_name: str = Field(min_length=1, max_length=50)
    sample_size: int = Field(ge=0)
    mean_brier_score: float = Field(ge=0.0, le=1.0)
    calibration_buckets: dict[str, Any] = Field(default_factory=dict)


class CalibrationCreate(CalibrationBase):
    pass


class CalibrationRead(CalibrationBase):
    id: UUID
    calculated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# StrategyNote
# ============================================================================
class StrategyNoteBase(BaseModel):
    goal_id: UUID
    strategy_id: UUID
    hypothesis: str = Field(min_length=5)
    reflection: str | None = None
    status: StrategyNoteStatus = StrategyNoteStatus.ACTIVE


class StrategyNoteCreate(StrategyNoteBase):
    pass


class StrategyNoteRead(StrategyNoteBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
