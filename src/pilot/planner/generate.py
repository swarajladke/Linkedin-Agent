"""Candidate action generation for Pilot Planner."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    Application,
    Company,
    Goal,
    Role,
    RoleAssessment,
    RoleStatus,
)
from pilot.planner.schemas import (
    CandidateAction,
    Diagnosis,
    DiagnosisCategory,
    RootCauseKind,
)

DEFAULT_FIT_FLOOR = 0.50
ELEVATED_FIT_FLOOR = 0.75


def generate_actions(
    session: Session,
    goal: Goal,
    diagnosis: Diagnosis,
    policy: dict | None = None,
) -> list[CandidateAction]:
    """
    Generate candidate application package actions over open, assessed roles.

    Enforces:
    - Exactly one action type: 'generate_application_package'.
    - Excludes roles with an existing Application or unexecuted/pending Action.
    - Low-fit targeting diagnosis raises the fit-score floor from baseline to elevated floor.
    - Blocked or constraint-conflicted goals produce 0 candidate actions.
    - Specific reason naming the diagnosis and assessment evidence claims.
    """
    if diagnosis.category in (DiagnosisCategory.BLOCKED, DiagnosisCategory.CONSTRAINT_CONFLICT):
        return []

    # Read policy parameters
    pol = policy
    if pol is None:
        active_strat = None
        strategies = getattr(goal, "strategies", []) or []
        for s in sorted(strategies, key=lambda x: getattr(x, "version", 0), reverse=True):
            if getattr(s, "retired_at", None) is None:
                active_strat = s
                break
        if active_strat is None and strategies:
            active_strat = strategies[0]
        pol = getattr(active_strat, "policy", {}) if active_strat else {}

    base_floor = float(pol.get("fit_floor", 0.50))
    elevated_floor = float(pol.get("elevated_fit_floor", 0.75))

    # Dynamic fit floor based on diagnosis
    fit_floor = base_floor
    is_low_fit_starved = any(
        h.cause == RootCauseKind.LOW_FIT_TARGETING for h in diagnosis.hypotheses
    )
    if is_low_fit_starved:
        fit_floor = elevated_floor

    # Existing applications for this goal
    applied_role_ids = set(
        session.scalars(select(Application.role_id).where(Application.goal_id == goal.id)).all()
    )

    # Existing pending or completed actions for this goal on roles
    actioned_role_ids = set(
        session.scalars(
            select(Action.target_id).where(
                Action.goal_id == goal.id,
                Action.target_type == "role",
                Action.action_type == "generate_application_package",
            )
        ).all()
    )

    excluded_role_ids = applied_role_ids | actioned_role_ids

    # Query open, assessed roles
    stmt = (
        select(Role, RoleAssessment, Company)
        .join(RoleAssessment, RoleAssessment.role_id == Role.id)
        .join(Company, Company.id == Role.company_id)
        .where(
            RoleAssessment.goal_id == goal.id,
            Role.status != RoleStatus.CLOSED,
            RoleAssessment.fit_score >= fit_floor,
        )
    )
    if excluded_role_ids:
        stmt = stmt.where(~Role.id.in_(excluded_role_ids))

    results = session.execute(stmt).all()

    candidates: list[CandidateAction] = []
    for role, assessment, company in results:
        claim_ids = [str(cid) for cid in assessment.supporting_claim_ids]
        claim_ref = (
            f"claims: [{', '.join(claim_ids[:3])}{'...' if len(claim_ids) > 3 else ''}]"
            if claim_ids
            else "no claims"
        )

        starved_desc = (
            f"starved({diagnosis.starved_stage})"
            if diagnosis.starved_stage
            else diagnosis.category.value
        )
        reason = (
            f"Generate application package for '{role.title}' at {company.name} "
            f"under diagnosis '{starved_desc}' with fit score {assessment.fit_score:.2f} "
            f"backed by {len(claim_ids)} verified claim(s) ({claim_ref})."
        )

        candidates.append(
            CandidateAction(
                action_type="generate_application_package",
                target_type="role",
                target_id=role.id,
                role_id=role.id,
                role_title=role.title,
                company_name=company.name,
                fit_score=assessment.fit_score,
                supporting_claim_ids=claim_ids,
                skill_gaps=assessment.skill_gaps,
                reason=reason,
            )
        )

    # Deterministic sort: fit_score descending, then role_id ascending
    candidates.sort(key=lambda c: (-c.fit_score, str(c.role_id)))
    return candidates
