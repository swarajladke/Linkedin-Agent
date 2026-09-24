"""Role intelligence assessment module."""

from pilot.intelligence.assessor import (
    RoleAssessor,
    compute_fit_score,
    derive_recommended_action,
)
from pilot.intelligence.repository import upsert_assessments
from pilot.schemas.intelligence import (
    RoleAssessmentBase,
    RoleAssessmentCreate,
    RoleAssessmentRead,
    SkillGap,
)

__all__ = [
    "RoleAssessmentBase",
    "RoleAssessmentCreate",
    "RoleAssessmentRead",
    "RoleAssessor",
    "SkillGap",
    "compute_fit_score",
    "derive_recommended_action",
    "upsert_assessments",
]
