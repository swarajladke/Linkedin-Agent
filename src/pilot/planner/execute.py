"""Execution engine for Pilot Planner actions integrated with the Critic verification gate."""

import json
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.critic import (
    Critic,
    CriticVerdict,
    build_voice_profile,
    get_user_voice_profile,
)
from pilot.db.models import (
    Action,
    Company,
    CriticReview,
    Escalation,
    EvidenceClaim,
    Goal,
    Role,
    RoleAssessment,
)
from pilot.extraction.llm import StructuredLLMClient
from pilot.planner.schemas import (
    DraftApplicationPackage,
    ExecutionResult,
)


def execute(
    session: Session,
    action: Action,
    *,
    now: datetime,
    llm: StructuredLLMClient | None = None,
) -> ExecutionResult:
    """
    Execute a single action of type 'generate_application_package' through the Critic gate.

    Guarantees:
    - Pre-execution invariant: action must have persisted ID and non-null prediction.
    - Draft packages are built from role assessment supporting claims.
    - Gate pipeline: Draft -> Critic Review -> (Pass / Regenerate / Drop).
    - Every attempt is persisted to critic_reviews table.
    - Max 2 regeneration attempts: failure on attempt 2 drops the action without Escalation.
    - Invariant: every Escalation payload traces to a critic_reviews row with verdict = 'pass'.
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
            f"Phase strictly permits only 'generate_application_package'."
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

    # 4. Resolve Candidate Voice Profile
    profile = None
    if user_id:
        try:
            profile = get_user_voice_profile(session, user_id)
        except Exception:
            profile = None
    if profile is None:
        profile = build_voice_profile([])

    # 5. Build Initial Draft Package (Attempt 1)
    bullets: list[str] = []
    for c in claims:
        bullets.append(f"• Demonstrates capability in: {c.claim}")

    if not bullets:
        bullets.append(f"• Role alignment verified against {role.title} requirements summary.")

    tailored_summary = f"Experienced software engineer aligned with {role.title} at {company_name}."

    draft_package = DraftApplicationPackage(
        role_id=role.id,
        role_title=role.title,
        company_name=company_name,
        tailored_summary=tailored_summary,
        tailored_bullets=bullets,
        supporting_claim_ids=[str(c.id) for c in claims],
    )

    critic = Critic(llm=llm)

    # Attempt 1 Review
    rev1 = critic.review(
        artifact=draft_package,
        claims=claims,
        profile=profile,
        role=role,
        attempt=1,
        action_id=action.id,
    )

    # Persist Attempt 1 review row
    review_row1 = CriticReview(
        action_id=action.id,
        attempt=1,
        artifact_text=rev1.artifact_text,
        grounding_passed=rev1.grounding_passed,
        voice_passed=rev1.voice_passed,
        factual_passed=rev1.factual_passed,
        verdict=rev1.verdict.value,
        failures=[f.model_dump() for f in rev1.failures],
        reviewed_at=now,
    )
    session.add(review_row1)
    session.flush()

    if rev1.verdict == CriticVerdict.PASS:
        # Pass on attempt 1 -> Escalate for human send
        draft_json = draft_package.model_dump_json()
        action.actual_outcome = draft_json
        action.executed_at = now
        action.outcome_at = now

        escalation_id = uuid.uuid4()
        escalation = Escalation(
            id=escalation_id,
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
            escalation_id=escalation_id,
            dropped=False,
            critic_verdict="pass",
            critic_attempts=1,
            role_title=role.title,
            company_name=company_name,
        )

    if rev1.verdict == CriticVerdict.REGENERATE:
        # Attempt 2: Redraft with failures as feedback
        blocking_failures = [f for f in rev1.failures if f.severity.value == "blocking"]
        offending_texts = [f.offending_text.lower() for f in blocking_failures if f.offending_text]
        repaired_bullets = [
            b for b in bullets if not any(off in b.lower() for off in offending_texts)
        ]
        if not repaired_bullets:
            repaired_bullets = [
                f"• Role alignment verified against {role.title} requirements summary."
            ]

        repaired_summary = (
            f"Experienced software engineer aligned with {role.title} at {company_name}."
        )

        repaired_package = DraftApplicationPackage(
            role_id=role.id,
            role_title=role.title,
            company_name=company_name,
            tailored_summary=repaired_summary,
            tailored_bullets=repaired_bullets,
            supporting_claim_ids=[str(c.id) for c in claims],
        )

        rev2 = critic.review(
            artifact=repaired_package,
            claims=claims,
            profile=profile,
            role=role,
            attempt=2,
            action_id=action.id,
        )

        # Persist Attempt 2 review row
        review_row2 = CriticReview(
            action_id=action.id,
            attempt=2,
            artifact_text=rev2.artifact_text,
            grounding_passed=rev2.grounding_passed,
            voice_passed=rev2.voice_passed,
            factual_passed=rev2.factual_passed,
            verdict=rev2.verdict.value,
            failures=[f.model_dump() for f in rev2.failures],
            reviewed_at=now,
        )
        session.add(review_row2)
        session.flush()

        if rev2.verdict == CriticVerdict.PASS:
            # Pass on attempt 2 -> Escalate for human send
            draft_json = repaired_package.model_dump_json()
            action.actual_outcome = draft_json
            action.executed_at = now
            action.outcome_at = now

            escalation_id = uuid.uuid4()
            escalation = Escalation(
                id=escalation_id,
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
                draft_package=repaired_package,
                escalation_id=escalation_id,
                dropped=False,
                critic_verdict="pass",
                critic_attempts=2,
                role_title=role.title,
                company_name=company_name,
            )

        # Failed attempt 2 -> Drop action (no Escalation)
        drop_record = {
            "status": "dropped",
            "reason": "critic_verification_failed",
            "final_verdict": rev2.verdict.value,
            "attempts": 2,
            "failures": [f.model_dump() for f in rev2.failures],
        }
        action.actual_outcome = json.dumps(drop_record)
        action.executed_at = now
        action.outcome_at = now

        return ExecutionResult(
            action_id=action.id,
            executed_at=now,
            draft_package=None,
            escalation_id=None,
            dropped=True,
            drop_reason="critic_verification_failed",
            critic_verdict="drop",
            critic_attempts=2,
            role_title=role.title,
            company_name=company_name,
        )

    # Initial verdict was DROP
    drop_record = {
        "status": "dropped",
        "reason": "critic_verification_failed",
        "final_verdict": rev1.verdict.value,
        "attempts": 1,
        "failures": [f.model_dump() for f in rev1.failures],
    }
    action.actual_outcome = json.dumps(drop_record)
    action.executed_at = now
    action.outcome_at = now

    return ExecutionResult(
        action_id=action.id,
        executed_at=now,
        draft_package=None,
        escalation_id=None,
        dropped=True,
        drop_reason="critic_verification_failed",
        critic_verdict="drop",
        critic_attempts=1,
        role_title=role.title,
        company_name=company_name,
    )
