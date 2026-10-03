"""Automated and timeout outcome resolution for Pilot actions."""

import json
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    ActionOutcome,
    Application,
    ApplicationStage,
    Goal,
)
from pilot.learning.schemas import ResolutionResult

# Stages indicating a positive outcome for an application action
POSITIVE_STAGES: set[ApplicationStage] = {
    ApplicationStage.SCREENING,
    ApplicationStage.TECHNICAL,
    ApplicationStage.FINAL,
    ApplicationStage.OFFER,
}

# Stages indicating an explicit terminal rejection
REJECTED_STAGES: set[ApplicationStage] = {
    ApplicationStage.REJECTED,
    ApplicationStage.WITHDRAWN,
}


def is_critic_dropped(action: Action) -> bool:
    """Check if action was dropped by Critic verification gate without human dispatch."""
    if not action.actual_outcome:
        return False
    try:
        data = json.loads(action.actual_outcome)
        if isinstance(data, dict) and data.get("status") == "dropped":
            return True
    except Exception:
        pass
    return False


def resolve_outcomes(
    session: Session,
    goal: Goal,
    *,
    now: datetime,
) -> ResolutionResult:
    """
    Resolve pending action predictions into binary outcomes with Brier score tracking.

    Rules:
    - Map each application stage transition to the action whose prediction it resolves.
    - Timeout resolution: an un-progressed action whose horizon has expired resolves as failure.
    - Critic drops are never resolved as funnel failures (they are a separate outcome class).
    - Idempotent: resolving twice produces zero duplicate rows.
    """
    # 1. Fetch all actions for this goal
    stmt = select(Action).where(Action.goal_id == goal.id).order_by(Action.created_at.asc())
    actions = session.execute(stmt).scalars().all()

    # 2. Pre-fetch applications for this goal mapped by role_id
    app_stmt = select(Application).where(Application.goal_id == goal.id)
    applications = session.execute(app_stmt).scalars().all()
    app_by_role: dict[str, Application] = {
        str(app.role_id): app for app in applications if app.role_id
    }

    result = ResolutionResult()

    for action in actions:
        # Skip actions that are already resolved
        if action.outcome is not None:
            continue

        # Skip critic dropped actions (never left the system)
        if is_critic_dropped(action):
            result.critic_drops_ignored += 1
            continue

        # Action must be executed to resolve
        if action.executed_at is None:
            result.open_predictions += 1
            continue

        p = float(action.predicted_probability) if action.predicted_probability is not None else 0.5
        role_key = str(action.target_id) if action.target_id else None
        app = app_by_role.get(role_key) if role_key else None

        # 3. Check for stage transition
        if app is not None and app.stage in POSITIVE_STAGES:
            brier = round((p - 1.0) ** 2, 4)
            outcome = ActionOutcome(
                action=action,
                action_id=action.id,
                binary_success=True,
                actual_outcome_details={
                    "source": "stage_transition",
                    "stage": app.stage.value,
                    "role_id": role_key,
                },
                brier_score=brier,
                diagnosis=f"Application advanced to positive stage '{app.stage.value}'",
                recorded_at=now,
            )
            action.outcome = outcome
            session.add(outcome)
            result.resolved_count += 1
            result.stage_transition_resolved += 1
            result.resolved_action_ids.append(action.id)
            continue

        if app is not None and app.stage in REJECTED_STAGES:
            brier = round((p - 0.0) ** 2, 4)
            outcome = ActionOutcome(
                action=action,
                action_id=action.id,
                binary_success=False,
                actual_outcome_details={
                    "source": "stage_transition",
                    "stage": app.stage.value,
                    "role_id": role_key,
                },
                brier_score=brier,
                diagnosis=f"Application closed with terminal stage '{app.stage.value}'",
                recorded_at=now,
            )
            action.outcome = outcome
            session.add(outcome)
            result.resolved_count += 1
            result.stage_transition_resolved += 1
            result.resolved_action_ids.append(action.id)
            continue

        # 4. Check for timeout resolution
        horizon = action.horizon_days if action.horizon_days is not None else 14
        deadline = action.executed_at + timedelta(days=horizon)

        if now >= deadline:
            brier = round((p - 0.0) ** 2, 4)
            outcome = ActionOutcome(
                action=action,
                action_id=action.id,
                binary_success=False,
                actual_outcome_details={
                    "source": "timeout",
                    "horizon_days": horizon,
                    "deadline": deadline.isoformat(),
                    "role_id": role_key,
                },
                brier_score=brier,
                diagnosis=f"Prediction horizon of {horizon} days expired without response",
                recorded_at=now,
            )
            action.outcome = outcome
            session.add(outcome)
            result.resolved_count += 1
            result.timeout_resolved += 1
            result.resolved_action_ids.append(action.id)
        else:
            result.open_predictions += 1

    session.flush()
    return result
