"""Pydantic schemas for Pilot Phase 5: Learning & Calibration."""

import enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Strategy Policy Schema
# ============================================================================
class StrategyPolicy(BaseModel):
    """Tunable policy configuration parameters governing Planner selection and priors."""

    fit_floor: float = Field(
        default=0.50, ge=0.0, le=1.0, description="Minimum role fit score floor"
    )
    elevated_fit_floor: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Elevated fit floor under low-fit starvation"
    )
    conversion_prior_factor: float = Field(
        default=0.40, ge=0.0, le=2.0, description="Multiplier converting fit score to prior"
    )
    default_conversion_prior: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Default uncalibrated baseline prior"
    )
    horizon_days: int = Field(
        default=14, ge=1, le=90, description="Prediction resolution horizon in days"
    )
    max_applications_per_day: int | None = Field(
        default=None, ge=1, description="Optional rate cap on daily applications"
    )
    urgency_multiplier_applications: float = Field(
        default=1.25, ge=0.5, le=3.0, description="Urgency bonus for starved applications"
    )
    urgency_multiplier_responses: float = Field(
        default=1.10, ge=0.5, le=3.0, description="Urgency bonus for starved responses"
    )
    urgency_multiplier_other: float = Field(
        default=1.05, ge=0.5, le=3.0, description="Urgency bonus for other starved stages"
    )


# ============================================================================
# Outcome Resolution Schemas
# ============================================================================
class ResolutionResult(BaseModel):
    """Aggregate result from resolving action outcomes."""

    resolved_count: int = 0
    stage_transition_resolved: int = 0
    timeout_resolved: int = 0
    critic_drops_ignored: int = 0
    open_predictions: int = 0
    resolved_action_ids: list[UUID] = Field(default_factory=list)


# ============================================================================
# Calibration Schemas
# ============================================================================
class DecileBucket(BaseModel):
    """Statistics for one predicted probability decile bucket [bin_lower, bin_upper)."""

    decile: int = Field(ge=1, le=10)
    bin_lower: float = Field(ge=0.0, le=1.0)
    bin_upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(default=0, ge=0)
    mean_predicted: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class MurphyDecomposition(BaseModel):
    """Murphy (1973) partition of Brier score: BS = Reliability - Resolution + Uncertainty."""

    reliability: float = Field(description="Distance from perfect calibration (lower is better)")
    resolution: float = Field(description="Distance of bin rates from base rate (higher is better)")
    uncertainty: float = Field(description="Climatological variance o_bar * (1 - o_bar)")
    brier_score: float = Field(description="Total mean Brier score")


class CalibrationReport(BaseModel):
    """Complete calibration report for a strategy version."""

    strategy_id: UUID
    sample_size: int = Field(ge=0)
    mean_brier_score: float = Field(ge=0.0)
    ece: float = Field(ge=0.0, le=1.0, description="Expected Calibration Error")
    brier_skill_score: float = Field(description="Skill score relative to climatological base rate")
    murphy_decomposition: MurphyDecomposition
    deciles: list[DecileBucket] = Field(default_factory=list)
    base_rate: float = Field(ge=0.0, le=1.0)
    beats_base_rate: bool = Field(description="True if Brier skill score > 0")
    summary: str


# ============================================================================
# Strategy Failure Detection, Reflection & Adoption Schemas
# ============================================================================
class FailureReasonKind(enum.StrEnum):
    CONSECUTIVE_STARVED = "consecutive_starved"
    CREDIBLE_INTERVAL_BREACHED = "credible_interval_breached"
    CALIBRATION_DRIFT = "calibration_drift"
    CRITIC_DROP_SPIKE = "critic_drop_spike"
    MISSED_DEADLINE = "missed_deadline"
    REFUTED_HYPOTHESIS = "refuted_hypothesis"


class FailureSignal(BaseModel):
    """Deterministic failure signal triggering strategy reflection."""

    kind: FailureReasonKind
    summary: str
    stage: str | None = None
    consecutive_cycles: int | None = None
    metric_value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyChange(BaseModel):
    """Bounded, validated alteration of a single policy parameter."""

    parameter: str
    old_value: float | int
    new_value: float | int
    delta: float | int


class Reflection(BaseModel):
    """Structured LLM reflection proposing validated policy change with falsifiable hypothesis."""

    signal: FailureSignal
    explanation: str
    hypothesis: str
    proposed_changes: list[PolicyChange] = Field(default_factory=list)


class AdoptionResult(BaseModel):
    """Result of evaluating and adopting a proposed strategy version."""

    adopted: bool
    rejection_reason: str | None = None
    old_version: int
    new_version: int | None = None
    new_strategy_id: UUID | None = None
    replay_diff: dict[str, Any] | None = None
