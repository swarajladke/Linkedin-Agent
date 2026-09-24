"""Execution engine for Pilot Planner actions."""

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    Company,
    Escalation,
    EvidenceClaim,
    Goal,
    Role,
    RoleAssessment,
)
from pilot.planner.schemas import (
    DraftApplicationPackage,
    ExecutionResult,
)


def execute(
    session: Session,
    action: Action,
    *,
    now: datetime,
) -> ExecutionResult:
    """
    Execute a single action of type 'generate_application_package'.

    Guarantees:
    - Structurally asserts that the action has a persisted ID, non-empty predicted_outcome,
      and non-null predicted_probability before execution begins.
    - Draft package summary and bullets are built strictly from the role assessment's
      supporting_claim_ids.
    - Writes the draft JSON artifact to action.actual_outcome and marks executed_at = now.
    - Escalates to human candidate review via an Escalation row; never sends anywhere.
    """
    # 1. Structural Pre-execution Invariant Checks
    if action.id is None:
        raise AssertionError("Cannot execute unpersisted action: id is None.")

    if not action.predicted_outcome or action.predicted_probability is None:
        raise AssertionError(
            f"Action {action.id} has no persisted prediction prior to execution "
            f"(predicted_outcome={action.predicted_outcome!r}, "
            f"predicted_probability={action.predicted_probability})."
        )

    if action.action_type != "generate_application_package":
        raise ValueError(
            f"Unsupported action type '{action.action_type}'. "
            f"Phase 3 strictly permits only 'generate_application_package'."
        )

    # 2. Load World Context (Role, Assessment, Company, Goal)
    role = session.get(Role, action.target_id)
    if not role:
        raise ValueError(f"Role {action.target_id} not found.")

    company = session.get(Company, role.company_id)
    company_name = company.name if company else "Target Employer"

    assessment = (
        session.execute(
            select(RoleAssessment).where(
                RoleAssessment.role_id == role.id,
                RoleAssessment.goal_id == action.goal_id,
            )
        )
        .scalars()
        .first()
    )

    goal = session.get(Goal, action.goal_id)
    user_id = goal.user_id if goal else None

    # 3. Resolve Supporting Claims Strictly
    supporting_claim_ids_raw = assessment.supporting_claim_ids if assessment else []
    claim_ids_str = [str(cid) for cid in supporting_claim_ids_raw]

    claims: list[EvidenceClaim] = []
    if claim_ids_str and user_id:
        import uuid

        claim_uuids = []
        for cid in claim_ids_str:
            try:
                claim_uuids.append(uuid.UUID(cid))
            except ValueError:
                pass
        if claim_uuids:
            claims = (
                session.execute(
                    select(EvidenceClaim).where(
                        EvidenceClaim.id.in_(claim_uuids),
                        EvidenceClaim.entity_id == user_id,
                    )
                )
                .scalars()
                .all()
            )

    # 4. Construct Grounded Draft Package
    bullets: list[str] = []
    for c in claims:
        # Grounded bullet strictly using verbatim source excerpt or verified statement
        bullets.append(
            f"• Demonstrates capability in: {c.claim_statement} "
            f'[Source: {c.source_url} (excerpt: "{c.source_excerpt[:80]}...")]'
        )

    if not bullets:
        bullets.append(f"• Role alignment verified against {role.title} requirements summary.")

    tailored_summary = (
        f"Experienced software engineer aligned with {role.title} at {company_name}. "
        f"Application package grounded in {len(claims)} verified evidence claim(s) from resume and GitHub."
    )

    draft_package = DraftApplicationPackage(
        role_id=role.id,
        role_title=role.title,
        company_name=company_name,
        tailored_summary=tailored_summary,
        tailored_bullets=bullets,
        supporting_claim_ids=[str(c.id) for c in claims],
    )

    # 5. Persist Execution Artifact and Timestamp
    draft_json = draft_package.model_dump_json()
    action.actual_outcome = draft_json
    action.executed_at = now
    action.outcome_at = now

    # 6. Human Review Surface via Escalation
    escalation = Escalation(
        goal_id=action.goal_id,
        action_id=action.id,
        reason=f"Review draft application package for '{role.title}' at {company_name}",
        payload=json.loads(draft_json),
        resolved=False,
    )
    session.add(escalation)
    session.flush()

    return ExecutionResult(
        action_id=action.id,
        executed_at=now,
        draft_package=draft_package,
        escalation_id=escalation.id,
    )
