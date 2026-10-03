"""Unit tests for outcome resolution, stage transitions, timeouts, and critic drop handling."""

import json
from datetime import UTC, datetime, timedelta

from pilot.db.models import (
    Action,
    Application,
    ApplicationStage,
    Company,
    Goal,
    Role,
    Strategy,
    User,
)
from pilot.learning.outcomes import resolve_outcomes


def test_stage_transition_resolution(db_session):
    """Application advancing to SCREENING resolves action as binary_success=True."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    user = User(email="outcome@example.com", full_name="Outcome User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Stage Transition Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    company = Company(name="Test Co 1")
    db_session.add(company)
    db_session.flush()

    role = Role(
        company_id=company.id,
        title="ML Engineer",
        location_type="remote",
        source="ashby",
        external_id="ext_role_1",
    )
    db_session.add(role)
    db_session.flush()

    app = Application(
        goal_id=goal.id,
        role_id=role.id,
        stage=ApplicationStage.SCREENING,
    )
    db_session.add(app)
    db_session.flush()

    action = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=role.id,
        reason="Test action",
        predicted_outcome="Interview response",
        predicted_probability=0.75,
        executed_at=now - timedelta(days=2),
        horizon_days=14,
    )
    db_session.add(action)
    db_session.flush()

    res = resolve_outcomes(db_session, goal, now=now)

    assert res.resolved_count == 1
    assert res.stage_transition_resolved == 1
    assert action.outcome is not None
    assert action.outcome.binary_success is True
    assert round(action.outcome.brier_score, 4) == round((0.75 - 1.0) ** 2, 4)


def test_stage_transition_rejection(db_session):
    """Application marked as REJECTED resolves action as binary_success=False."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    user = User(email="reject@example.com", full_name="Reject User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Reject Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    company = Company(name="Test Co 2")
    db_session.add(company)
    db_session.flush()

    role = Role(
        company_id=company.id,
        title="AI Engineer",
        location_type="remote",
        source="greenhouse",
        external_id="ext_role_2",
    )
    db_session.add(role)
    db_session.flush()

    app = Application(
        goal_id=goal.id,
        role_id=role.id,
        stage=ApplicationStage.REJECTED,
    )
    db_session.add(app)
    db_session.flush()

    action = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=role.id,
        reason="Test rejection",
        predicted_outcome="Interview response",
        predicted_probability=0.60,
        executed_at=now - timedelta(days=3),
        horizon_days=14,
    )
    db_session.add(action)
    db_session.flush()

    res = resolve_outcomes(db_session, goal, now=now)

    assert res.resolved_count == 1
    assert res.stage_transition_resolved == 1
    assert action.outcome is not None
    assert action.outcome.binary_success is False
    assert round(action.outcome.brier_score, 4) == round((0.60 - 0.0) ** 2, 4)


def test_timeout_resolution_expired_vs_unexpired(db_session):
    """An action whose horizon expired resolves as failure; unexpired stays open."""
    now = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    user = User(email="timeout@example.com", full_name="Timeout User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Timeout Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    company = Company(name="Test Co 3")
    db_session.add(company)
    db_session.flush()

    role1 = Role(
        company_id=company.id,
        title="Role 1",
        location_type="remote",
        source="ashby",
        external_id="t_1",
    )
    role2 = Role(
        company_id=company.id,
        title="Role 2",
        location_type="remote",
        source="ashby",
        external_id="t_2",
    )
    db_session.add_all([role1, role2])
    db_session.flush()

    # Action 1: executed 20 days ago, horizon 14 -> Expired!
    action_expired = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=role1.id,
        reason="Expired action",
        predicted_outcome="Interview response",
        predicted_probability=0.70,
        executed_at=now - timedelta(days=20),
        horizon_days=14,
    )
    # Action 2: executed 5 days ago, horizon 14 -> Not expired!
    action_open = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=role2.id,
        reason="Open action",
        predicted_outcome="Interview response",
        predicted_probability=0.70,
        executed_at=now - timedelta(days=5),
        horizon_days=14,
    )
    db_session.add_all([action_expired, action_open])
    db_session.flush()

    res = resolve_outcomes(db_session, goal, now=now)

    assert res.resolved_count == 1
    assert res.timeout_resolved == 1
    assert res.open_predictions == 1
    assert action_expired.outcome is not None
    assert action_expired.outcome.binary_success is False
    assert action_open.outcome is None


def test_critic_drops_are_ignored(db_session):
    """Critic drops are never resolved as funnel failures."""
    now = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    user = User(email="drop@example.com", full_name="Drop User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Drop Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    dropped_action = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        reason="Dropped action",
        predicted_outcome="Interview response",
        predicted_probability=0.80,
        executed_at=now - timedelta(days=20),
        horizon_days=14,
        actual_outcome=json.dumps({"status": "dropped", "reason": "Grounding failed twice"}),
    )
    db_session.add(dropped_action)
    db_session.flush()

    res = resolve_outcomes(db_session, goal, now=now)

    assert res.resolved_count == 0
    assert res.critic_drops_ignored == 1
    assert dropped_action.outcome is None


def test_resolution_idempotency(db_session):
    """Running resolve_outcomes twice does not create duplicate outcome rows."""
    now = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    user = User(email="idem_out@example.com", full_name="Idem Out User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Idem Outcome Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    action = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        reason="Test idempotency",
        predicted_outcome="Interview response",
        predicted_probability=0.70,
        executed_at=now - timedelta(days=20),
        horizon_days=14,
    )
    db_session.add(action)
    db_session.flush()

    res1 = resolve_outcomes(db_session, goal, now=now)
    assert res1.resolved_count == 1
    assert action.outcome is not None

    res2 = resolve_outcomes(db_session, goal, now=now)
    assert res2.resolved_count == 0
    assert res2.open_predictions == 0
