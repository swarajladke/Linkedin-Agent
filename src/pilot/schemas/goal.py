"""Pydantic schemas for Goals, Target Specs, and Sub-goals."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from pilot.db.models import GoalStatus


class TargetSpec(BaseModel):
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    unstated_but_real: list[str] = Field(
        default_factory=list,
        description="Implicit baseline requirements inferred for this role",
    )


class SubGoal(BaseModel):
    id: str = Field(description="Sub-goal identifier e.g. sg_1")
    title: str = Field(min_length=1)
    metric_target: dict[str, Any] = Field(
        default_factory=dict,
        description="Numeric target e.g. {'applications_submitted': 10}",
    )
    deadline: datetime = Field(description="Back-solved target completion timestamp")
    status: str = Field(
        default="pending", description="pending | in_progress | achieved | breached"
    )
    depends_on: list[str] = Field(default_factory=list)


class GoalBase(BaseModel):
    objective_text: str = Field(min_length=5)
    constraints_json: dict[str, Any] = Field(default_factory=dict)
    target_spec: TargetSpec
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    sub_goals: list[SubGoal] = Field(default_factory=list)
    deadline: datetime
    status: GoalStatus = GoalStatus.ACTIVE


class GoalCreate(GoalBase):
    user_id: UUID


class GoalRead(GoalBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
