"""Pydantic schemas for the Pilot Planner decision loop."""

import enum
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Funnel & Observation Schemas
# ============================================================================
class FunnelStageCounts(BaseModel):
    """Aggregate volume counts across each stage of the conversion funnel."""

    roles_sourced: int = Field(default=0, ge=0)
    roles_assessed: int = Field(default=0, ge=0)
    applications: int = Field(default=0, ge=0)
    responses: int = Field(default=0, ge=0)
    interviews: int = Field(default=0, ge=0)
    offers: int = Field(default=0, ge=0)


class ConversionRates(BaseModel):
    """Stage-to-stage conversion rates (bounded in [0.0, 1.0])."""

    sourcing_to_assessed: float = Field(default=0.0, ge=0.0, le=1.0)
    assessed_to_applied: float = Field(default=0.0, ge=0.0, le=1.0)
    applied_to_response: float = Field(default=0.0, ge=0.0, le=1.0)
    response_to_interview: float = Field(default=0.0, ge=0.0, le=1.0)
    interview_to_offer: float = Field(default=0.0, ge=0.0, le=1.0)


class TimelineProgress(BaseModel):
    """Elapsed and remaining time relative to the goal's deadline."""

    elapsed_days: float = Field(default=0.0, ge=0.0)
    remaining_days: float = Field(default=0.0, ge=0.0)
    total_days: float = Field(default=0.0, ge=0.0)
    pace_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class SubGoalProgress(BaseModel):
    """Tracking progress against an individual compiled sub-goal milestone."""

    id: str
    title: str
    metric_target: dict[str, Any] = Field(default_factory=dict)
    actual_value: float = 0.0
    target_value: float = 0.0
    is_achieved: bool = False
    deadline: datetime | None = None


class Observation(BaseModel):
    """Deterministic, pure world-state observation for a goal."""

    goal_id: UUID
    observed_at: datetime
    funnel_counts: FunnelStageCounts
    conversion_rates: ConversionRates
    timeline: TimelineProgress
    sub_goal_progress: list[SubGoalProgress] = Field(default_factory=list)
    delta_since_last_cycle: dict[str, int] = Field(default_factory=dict)
    last_cycle_number: int | None = None
    evidence_claims_count: int = 0
    active_escalations_count: int = 0
    average_fit_score: float | None = None


# ============================================================================
# Diagnosis Schemas
# ============================================================================
class DiagnosisCategory(enum.StrEnum):
    ON_PACE = "on_pace"
    STARVED = "starved"
    BLOCKED = "blocked"
    CONSTRAINT_CONFLICT = "constraint_conflict"


class RootCauseKind(enum.StrEnum):
    LOW_FIT_TARGETING = "low_fit_targeting"
    WEAK_EVIDENCE_COVERAGE = "weak_evidence_coverage"
    INSUFFICIENT_VOLUME = "insufficient_volume"
    CONSTRAINT_TOO_NARROW = "constraint_too_narrow"
    TIMING = "timing"


class RootCauseHypothesis(BaseModel):
    """Structured hypothesis explaining a starved funnel stage, backed by observation numbers."""

    cause: RootCauseKind
    explanation: str
    supporting_metric_name: str
    supporting_metric_value: float


class Diagnosis(BaseModel):
    """Diagnostic state of the goal pursue loop."""

    category: DiagnosisCategory
    starved_stage: str | None = None
    blocked_reason: str | None = None
    conflict_details: str | None = None
    stage_health: dict[str, str] = Field(default_factory=dict)
    hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)
    insufficient_data: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_on_pace(self) -> bool:
        return self.category == DiagnosisCategory.ON_PACE

    @property
    def is_starved(self) -> bool:
        return self.category == DiagnosisCategory.STARVED

    @property
    def is_blocked(self) -> bool:
        return self.category == DiagnosisCategory.BLOCKED

    @property
    def is_constraint_conflict(self) -> bool:
        return self.category == DiagnosisCategory.CONSTRAINT_CONFLICT


# ============================================================================
# Candidate & Scored Action Schemas
# ============================================================================
class CandidateAction(BaseModel):
    """Candidate action proposal generated for an assessed role."""

    action_type: str = "generate_application_package"
    target_type: str = "role"
    target_id: UUID
    role_id: UUID
    role_title: str
    company_name: str
    fit_score: float
    supporting_claim_ids: list[str] = Field(default_factory=list)
    skill_gaps: list[dict[str, Any]] = Field(default_factory=list)
    reason: str


class ScoredAction(BaseModel):
    """Action scored by deterministic expected value function."""

    candidate: CandidateAction
    score: float
    urgency: float
    cost: Decimal = Decimal("0.0")
    predicted_probability: float
    predicted_outcome: str


# ============================================================================
# Execution & Replay Results
# ============================================================================
class DraftApplicationPackage(BaseModel):
    """Application materials draft produced strictly from grounded evidence claims."""

    role_id: UUID
    role_title: str
    company_name: str
    tailored_summary: str
    tailored_bullets: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Result of executing an action."""

    action_id: UUID
    executed_at: datetime
    draft_package: DraftApplicationPackage
    escalation_id: UUID


class CycleResult(BaseModel):
    """Result of orchestrating a full decision cycle."""

    cycle_id: UUID | None = None
    cycle_number: int
    started_at: datetime
    completed_at: datetime
    dry_run: bool
    observation: Observation
    diagnosis: Diagnosis
    actions_proposed: int
    actions_selected: int
    selected_actions: list[ScoredAction] = Field(default_factory=list)
    execution_results: list[ExecutionResult] = Field(default_factory=list)


class ReplayResult(BaseModel):
    """Diff comparing historical actions selected vs counterfactual selection."""

    cycle_id: UUID
    cycle_number: int
    original_action_ids: list[UUID] = Field(default_factory=list)
    counterfactual_action_ids: list[UUID] = Field(default_factory=list)
    added_action_ids: list[UUID] = Field(default_factory=list)
    removed_action_ids: list[UUID] = Field(default_factory=list)
    rationale_diff: str
