"""Repository functions for versioned role assessments."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import RoleAssessment
from pilot.schemas.intelligence import RoleAssessmentCreate


def upsert_assessments(
    session: Session,
    assessments: list[RoleAssessmentCreate],
) -> list[RoleAssessment]:
    """
    Upsert role assessments on unique key (role_id, goal_id, assessor_version).

    - Re-assessing under the same assessor_version updates in place.
    - Bumping assessor_version creates a new assessment row and preserves prior runs for replay.
    """
    now = datetime.now(UTC)
    persisted: list[RoleAssessment] = []

    for a in assessments:
        stmt = select(RoleAssessment).where(
            RoleAssessment.role_id == a.role_id,
            RoleAssessment.goal_id == a.goal_id,
            RoleAssessment.assessor_version == a.assessor_version,
        )
        existing = session.execute(stmt).scalars().first()

        # Serialize supporting claim IDs to strings for JSONB compatibility
        claim_ids_json = [str(cid) for cid in a.supporting_claim_ids]
        # Serialize skill gaps
        skill_gaps_json: list[dict[str, Any]] = [
            g.model_dump() if hasattr(g, "model_dump") else dict(g) for g in a.skill_gaps
        ]

        if existing:
            existing.fit_score = a.fit_score
            existing.fit_rationale = a.fit_rationale
            existing.supporting_claim_ids = claim_ids_json
            existing.skill_gaps = skill_gaps_json
            existing.company_context = a.company_context
            existing.recommended_action = a.recommended_action
            existing.assessed_at = a.assessed_at or now
            persisted.append(existing)
        else:
            new_assessment = RoleAssessment(
                role_id=a.role_id,
                goal_id=a.goal_id,
                fit_score=a.fit_score,
                fit_rationale=a.fit_rationale,
                supporting_claim_ids=claim_ids_json,
                skill_gaps=skill_gaps_json,
                company_context=a.company_context,
                recommended_action=a.recommended_action,
                assessed_at=a.assessed_at or now,
                assessor_version=a.assessor_version,
            )
            session.add(new_assessment)
            persisted.append(new_assessment)

    session.flush()
    return persisted
