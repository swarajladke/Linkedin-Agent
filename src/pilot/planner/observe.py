"""Deterministic observation of world model state for the Pilot Planner."""

from datetime import datetime

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Application,
    ApplicationStage,
    Cycle,
    Escalation,
    EvidenceClaim,
    Goal,
    Role,
    RoleAssessment,
    RoleStatus,
)
from pilot.planner.schemas import (
    ConversionRates,
    FunnelStageCounts,
    Observation,
    SubGoalProgress,
    TimelineProgress,
)

# Funnel stage membership sets
SUBMITTED_STAGES = {
    ApplicationStage.APPLIED,
    ApplicationStage.SCREENING,
    ApplicationStage.TECHNICAL,
    ApplicationStage.FINAL,
    ApplicationStage.OFFER,
    ApplicationStage.REJECTED,
    ApplicationStage.WITHDRAWN,
}

RESPONSE_STAGES = {
    ApplicationStage.SCREENING,
    ApplicationStage.TECHNICAL,
    ApplicationStage.FINAL,
    ApplicationStage.OFFER,
}

INTERVIEW_STAGES = {
    ApplicationStage.SCREENING,
    ApplicationStage.TECHNICAL,
    ApplicationStage.FINAL,
    ApplicationStage.OFFER,
}

OFFER_STAGES = {
    ApplicationStage.OFFER,
}


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 on zero-denominator, clamped to [0.0, 1.0] rounded to 4 decimals."""
    if denominator <= 0.0:
        return 0.0
    return round(min(1.0, max(0.0, float(numerator) / float(denominator))), 4)


def observe(session: Session, goal: Goal, *, now: datetime) -> Observation:
    """
    Pure read of world model state. No LLM calls.

    Computes:
    - Funnel counts by stage (roles sourced, assessed, applied, responses, interviews, offers)
    - Per-stage conversion rates (safe against zero denominators)
    - Days elapsed, remaining, total, and pace fraction
    - Sub-goal progress vs. compiled targets
    - Delta since previous cycle (if any)
    """
    # 1. Funnel Counts
    roles_sourced = (
        session.scalar(select(func.count(Role.id)).where(Role.status != RoleStatus.CLOSED)) or 0
    )

    roles_assessed = (
        session.scalar(
            select(func.count(distinct(RoleAssessment.role_id))).where(
                RoleAssessment.goal_id == goal.id
            )
        )
        or 0
    )

    applications_query = (
        session.execute(select(Application.stage).where(Application.goal_id == goal.id))
        .scalars()
        .all()
    )

    applications_count = sum(1 for s in applications_query if s in SUBMITTED_STAGES)
    responses_count = sum(1 for s in applications_query if s in RESPONSE_STAGES)
    interviews_count = sum(1 for s in applications_query if s in INTERVIEW_STAGES)
    offers_count = sum(1 for s in applications_query if s in OFFER_STAGES)

    funnel_counts = FunnelStageCounts(
        roles_sourced=roles_sourced,
        roles_assessed=roles_assessed,
        applications=applications_count,
        responses=responses_count,
        interviews=interviews_count,
        offers=offers_count,
    )

    # 2. Conversion Rates
    conversion_rates = ConversionRates(
        sourcing_to_assessed=_safe_div(roles_assessed, roles_sourced),
        assessed_to_applied=_safe_div(applications_count, roles_assessed),
        applied_to_response=_safe_div(responses_count, applications_count),
        response_to_interview=_safe_div(interviews_count, responses_count),
        interview_to_offer=_safe_div(offers_count, interviews_count),
    )

    # 3. Timeline Progress
    elapsed_seconds = max(0.0, (now - goal.created_at).total_seconds())
    total_seconds = max(1.0, (goal.deadline - goal.created_at).total_seconds())
    remaining_seconds = max(0.0, (goal.deadline - now).total_seconds())

    elapsed_days = elapsed_seconds / 86400.0
    total_days = total_seconds / 86400.0
    remaining_days = remaining_seconds / 86400.0
    pace_fraction = min(1.0, max(0.0, elapsed_seconds / total_seconds))

    timeline = TimelineProgress(
        elapsed_days=round(elapsed_days, 2),
        remaining_days=round(remaining_days, 2),
        total_days=round(total_days, 2),
        pace_fraction=round(pace_fraction, 4),
    )

    # 4. Evidence Claims & Active Escalations & Average Fit Score
    evidence_claims_count = (
        session.scalar(
            select(func.count(EvidenceClaim.id)).where(EvidenceClaim.entity_id == goal.user_id)
        )
        or 0
    )

    active_escalations_count = (
        session.scalar(
            select(func.count(Escalation.id)).where(
                Escalation.goal_id == goal.id,
                Escalation.resolved.is_(False),
            )
        )
        or 0
    )

    avg_fit = session.scalar(
        select(func.avg(RoleAssessment.fit_score)).where(RoleAssessment.goal_id == goal.id)
    )
    average_fit_score = float(avg_fit) if avg_fit is not None else None

    # 5. Sub-Goal Progress
    sub_goal_progress_list: list[SubGoalProgress] = []
    sub_goals_data = goal.sub_goals or []
    for sg_entry in sub_goals_data:
        if isinstance(sg_entry, dict):
            sg_id = str(sg_entry.get("id", ""))
            sg_title = str(sg_entry.get("title", ""))
            sg_target = dict(sg_entry.get("metric_target", {}))
            sg_deadline = sg_entry.get("deadline")
        else:
            sg_id = getattr(sg_entry, "id", "")
            sg_title = getattr(sg_entry, "title", "")
            sg_target = getattr(sg_entry, "metric_target", {})
            sg_deadline = getattr(sg_entry, "deadline", None)

        actual_val = 0.0
        target_val = 0.0

        if "evidence_claims_verified" in sg_target:
            target_val = float(sg_target["evidence_claims_verified"])
            actual_val = float(evidence_claims_count)
        elif "roles_identified" in sg_target:
            target_val = float(sg_target["roles_identified"])
            actual_val = float(roles_sourced)
        elif "applications_submitted" in sg_target:
            target_val = float(sg_target["applications_submitted"])
            actual_val = float(applications_count)
        elif "interviews_completed" in sg_target:
            target_val = float(sg_target["interviews_completed"])
            actual_val = float(interviews_count)
        elif "offers_received" in sg_target:
            target_val = float(sg_target["offers_received"])
            actual_val = float(offers_count)

        if isinstance(sg_deadline, str):
            try:
                parsed_deadline = datetime.fromisoformat(sg_deadline)
            except ValueError:
                parsed_deadline = None
        elif isinstance(sg_deadline, datetime):
            parsed_deadline = sg_deadline
        else:
            parsed_deadline = None

        sub_goal_progress_list.append(
            SubGoalProgress(
                id=sg_id,
                title=sg_title,
                metric_target=sg_target,
                actual_value=actual_val,
                target_value=target_val,
                is_achieved=(actual_val >= target_val if target_val > 0 else False),
                deadline=parsed_deadline,
            )
        )

    # 6. Delta since Previous Cycle
    prev_cycle_stmt = (
        select(Cycle).where(Cycle.goal_id == goal.id).order_by(Cycle.cycle_number.desc()).limit(1)
    )
    prev_cycle = session.execute(prev_cycle_stmt).scalars().first()

    deltas: dict[str, int] = {}
    last_cycle_number: int | None = None

    if prev_cycle and prev_cycle.observation:
        last_cycle_number = prev_cycle.cycle_number
        prev_obs = prev_cycle.observation
        prev_counts = prev_obs.get("funnel_counts", {})
        deltas = {
            "roles_sourced": roles_sourced - int(prev_counts.get("roles_sourced", 0)),
            "roles_assessed": roles_assessed - int(prev_counts.get("roles_assessed", 0)),
            "applications": applications_count - int(prev_counts.get("applications", 0)),
            "responses": responses_count - int(prev_counts.get("responses", 0)),
            "interviews": interviews_count - int(prev_counts.get("interviews", 0)),
            "offers": offers_count - int(prev_counts.get("offers", 0)),
        }
    else:
        deltas = {
            "roles_sourced": roles_sourced,
            "roles_assessed": roles_assessed,
            "applications": applications_count,
            "responses": responses_count,
            "interviews": interviews_count,
            "offers": offers_count,
        }

    return Observation(
        goal_id=goal.id,
        observed_at=now,
        funnel_counts=funnel_counts,
        conversion_rates=conversion_rates,
        timeline=timeline,
        sub_goal_progress=sub_goal_progress_list,
        delta_since_last_cycle=deltas,
        last_cycle_number=last_cycle_number,
        evidence_claims_count=evidence_claims_count,
        active_escalations_count=active_escalations_count,
        average_fit_score=average_fit_score,
    )
