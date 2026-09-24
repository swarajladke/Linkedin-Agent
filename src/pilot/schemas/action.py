"""Pydantic schemas for Actions, Action Outcomes, Funnel-Level Predictions, and Escalations."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Action
# ============================================================================
class ActionBase(BaseModel):
    goal_id: UUID
    strategy_id: UUID
    action_type: str = Field(min_length=1, max_length=100)
    target_type: str = Field(
        min_length=1,
        max_length=50,
        description="Discriminator: role | person | application | system",
    )
    target_id: UUID | None = None
    reason: str = Field(min_length=5, description="Auditable reasoning trace")
    predicted_outcome: str = Field(min_length=5, description="Falsifiable prediction statement")
    predicted_probability: float = Field(ge=0.0, le=1.0)
    cost: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"))
    strategy_version: str = Field(min_length=1, max_length=50)
    executed_at: datetime | None = None
    actual_outcome: str | None = None
    outcome_at: datetime | None = None


class ActionCreate(ActionBase):
    pass


class ActionRead(ActionBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# ActionOutcome
# ============================================================================
class ActionOutcomeBase(BaseModel):
    action_id: UUID
    binary_success: bool
    actual_outcome_details: dict[str, Any] | None = None
    diagnosis: str | None = None


class ActionOutcomeCreate(ActionOutcomeBase):
    pass


class ActionOutcomeRead(ActionOutcomeBase):
    id: UUID
    brier_score: float = Field(ge=0.0, le=1.0)
    recorded_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Prediction (Funnel-Level Only)
# ============================================================================
class PredictionBase(BaseModel):
    """Funnel-level probabilistic forecasts only (not per-action predictions)."""

    goal_id: UUID
    claim_statement: str = Field(min_length=5)
    predicted_probability: float = Field(ge=0.0, le=1.0)
    resolution_deadline: datetime | None = None
    is_resolved: bool = False
    outcome_truth: bool | None = None


class PredictionCreate(PredictionBase):
    pass


class PredictionRead(PredictionBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Escalation
# ============================================================================
class EscalationBase(BaseModel):
    goal_id: UUID
    action_id: UUID | None = None
    reason: str = Field(min_length=5)
    payload: dict[str, Any] | None = None
    resolved: bool = False
    resolved_at: datetime | None = None
    resolution_notes: str | None = None


class EscalationCreate(EscalationBase):
    pass


class EscalationRead(EscalationBase):
    id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
