"""Schemas for role assessment, skill gaps, and intelligence evaluation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SkillGap(BaseModel):
    """Identified requirement gap between candidate evidence and role specification."""

    gap: str
    severity: str = Field(default="medium")  # low, medium, high
    blocking: bool = False


class RoleAssessmentBase(BaseModel):
    """Core attributes of a role assessment against a career goal."""

    role_id: UUID
    goal_id: UUID
    fit_score: float = Field(ge=0.0, le=1.0)
    fit_rationale: str
    supporting_claim_ids: list[UUID] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    company_context: dict[str, object] = Field(default_factory=dict)
    recommended_action: str = Field(max_length=50)
    assessor_version: str = Field(default="v1", max_length=50)


class RoleAssessmentCreate(RoleAssessmentBase):
    """Payload for creating or updating a role assessment."""

    assessed_at: datetime | None = None


class RoleAssessmentRead(RoleAssessmentBase):
    """Persisted role assessment entity."""

    id: UUID
    assessed_at: datetime

    model_config = ConfigDict(from_attributes=True)
