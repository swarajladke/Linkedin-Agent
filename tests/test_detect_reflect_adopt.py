"""Unit tests for failure detection, bounded reflection, and replay-gated adoption."""

from datetime import UTC, datetime, timedelta

import pytest

from pilot.db.models import (
    Action,
    ActionOutcome,
    Cycle,
    Goal,
    Strategy,
    User,
)
from pilot.learning.adopt import evaluate_and_adopt, rollback_strategy
from pilot.learning.detect import detect_strategy_failure
from pilot.learning.reflect import (
    RawPolicyChangeProposal,
    reflect,
    validate_and_build_policy_changes,
)
from pilot.learning.schemas import FailureReasonKind, FailureSignal


def test_detect_returns_none_below_sample_floor(db_session):
    """Statistical failure triggers return None below minimum sample size (no pivoting on noise)."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    user = User(email="noise@example.com", full_name="Noise User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Noise Test Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={"default_conversion_prior": 0.70})
    db_session.add(strategy)
    db_session.flush()

    # Only 2 resolved failures (less than min_sample_size=5)
    for i in range(2):
        action = Action(
            goal_id=goal.id,
            strategy_id=strategy.id,
            action_type="generate_application_package",
            target_type="role",
            reason=f"Action {i}",
            predicted_outcome="Outcome",
            predicted_probability=0.70,
            executed_at=now - timedelta(days=5),
        )
        db_session.add(action)
        db_session.flush()
        outcome = ActionOutcome(
            action=action,
            action_id=action.id,
            binary_success=False,
            brier_score=0.49,
            recorded_at=now,
        )
        db_session.add(outcome)
    db_session.flush()

    sig = detect_strategy_failure(db_session, goal, strategy, min_sample_size=5, now=now)
    assert sig is None


def test_detect_fires_on_k_consecutive_starved_cycles(db_session):
    """Detection fires when the same funnel stage is diagnosed starved for K consecutive cycles."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    user = User(email="starve@example.com", full_name="Starve User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Starve Goal",
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

    # 3 consecutive cycles diagnosed starved at 'applications'
    for c_num in range(1, 4):
        cycle = Cycle(
            goal_id=goal.id,
            cycle_number=c_num,
            strategy_id=strategy.id,
            observation={},
            diagnosis={"category": "starved", "starved_stage": "applications"},
            actions_proposed=0,
            actions_selected=0,
            started_at=now - timedelta(days=4 - c_num),
            completed_at=now - timedelta(days=4 - c_num),
        )
        db_session.add(cycle)
    db_session.flush()

    sig = detect_strategy_failure(
        db_session, goal, strategy, consecutive_starved_threshold=3, now=now
    )
    assert sig is not None
    assert sig.kind == FailureReasonKind.CONSECUTIVE_STARVED
    assert sig.stage == "applications"
    assert sig.consecutive_cycles == 3


def test_reflect_rejects_unallowed_parameter_and_unbounded_deltas():
    """Reflection rejects parameters outside allow-list and deltas exceeding bounds."""
    current_policy = {"fit_floor": 0.50, "conversion_prior_factor": 0.40}

    # 1. Parameter outside allow-list -> rejected
    unallowed = [RawPolicyChangeProposal(parameter="magic_bonus", new_value=100.0)]
    with pytest.raises(ValueError, match="outside the allowed tunable policy parameters"):
        validate_and_build_policy_changes(current_policy, unallowed)

    # 2. Delta > 0.15 on fit_floor (0.50 -> 0.70, delta = 0.20) -> rejected
    unbounded_delta = [RawPolicyChangeProposal(parameter="fit_floor", new_value=0.70)]
    with pytest.raises(ValueError, match="exceeds maximum allowed delta"):
        validate_and_build_policy_changes(current_policy, unbounded_delta)

    # 3. Value outside bounds (fit_floor max is 0.90) -> rejected
    out_of_bounds = [RawPolicyChangeProposal(parameter="fit_floor", new_value=0.95)]
    with pytest.raises(ValueError):
        validate_and_build_policy_changes({"fit_floor": 0.85}, out_of_bounds)

    # 4. More than 2 changes proposed -> rejected
    too_many = [
        RawPolicyChangeProposal(parameter="fit_floor", new_value=0.55),
        RawPolicyChangeProposal(parameter="conversion_prior_factor", new_value=0.45),
        RawPolicyChangeProposal(parameter="default_conversion_prior", new_value=0.65),
    ]
    with pytest.raises(ValueError, match="exceeding the limit"):
        validate_and_build_policy_changes(current_policy, too_many)

    # 5. Valid bounded change -> accepted
    valid = [RawPolicyChangeProposal(parameter="fit_floor", new_value=0.60)]
    changes = validate_and_build_policy_changes(current_policy, valid)
    assert len(changes) == 1
    assert changes[0].parameter == "fit_floor"
    assert changes[0].delta == 0.10


def test_adoption_rejected_when_replay_selection_is_identical(db_session):
    """Adoption is rejected when the counterfactual replay selection is identical."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    user = User(email="identical@example.com", full_name="Identical User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Replay Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(
        goal_id=goal.id,
        version=1,
        policy={"fit_floor": 0.50, "conversion_prior_factor": 0.40},
    )
    db_session.add(strategy)
    db_session.flush()

    # 3 cycles under strategy 1 (satisfying cooldown=3)
    for c_num in range(1, 4):
        c = Cycle(
            goal_id=goal.id,
            cycle_number=c_num,
            strategy_id=strategy.id,
            observation={},
            diagnosis={"category": "on_pace"},
            started_at=now - timedelta(days=4 - c_num),
            completed_at=now - timedelta(days=4 - c_num),
        )
        db_session.add(c)
    db_session.flush()

    # Reflection with a tiny change that doesn't change selection (no roles available)
    sig = FailureSignal(
        kind=FailureReasonKind.CONSECUTIVE_STARVED,
        summary="Starved",
        stage="applications",
    )
    refl = reflect(goal, strategy, sig, [])

    result = evaluate_and_adopt(
        db_session,
        goal,
        refl,
        lookback_cycles=3,
        cooldown_cycles=3,
        now=now,
    )

    assert result.adopted is False
    assert "zero selection differences" in (result.rejection_reason or "")


def test_cooldown_blocks_second_adoption_within_m_cycles(db_session):
    """A new adoption is rejected if current strategy has run fewer than M cycles."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    user = User(email="cool@example.com", full_name="Cool User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Cool Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(
        goal_id=goal.id,
        version=1,
        policy={"fit_floor": 0.50},
    )
    db_session.add(strategy)
    db_session.flush()

    # Only 1 cycle completed under strategy v1
    c = Cycle(
        goal_id=goal.id,
        cycle_number=1,
        strategy_id=strategy.id,
        observation={},
        diagnosis={"category": "on_pace"},
        started_at=now - timedelta(days=1),
        completed_at=now - timedelta(days=1),
    )
    db_session.add(c)
    db_session.flush()

    sig = FailureSignal(kind=FailureReasonKind.CREDIBLE_INTERVAL_BREACHED, summary="Breach")
    refl = reflect(goal, strategy, sig, [])

    # Evaluate adoption with cooldown_cycles = 3
    result = evaluate_and_adopt(db_session, goal, refl, cooldown_cycles=3, now=now)
    assert result.adopted is False
    assert "Cooldown active" in (result.rejection_reason or "")


def test_rollback_strategy_creates_new_version_preserving_history(db_session):
    """Rollback creates a new version v+1 copying historical policy without mutating history."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    user = User(email="roll@example.com", full_name="Roll User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Rollback Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    # Version 1 policy: fit_floor = 0.50
    v1 = Strategy(
        goal_id=goal.id,
        version=1,
        policy={"fit_floor": 0.50, "conversion_prior_factor": 0.40},
        created_at=now - timedelta(days=10),
        retired_at=now - timedelta(days=5),
    )
    # Version 2 policy: fit_floor = 0.65
    v2 = Strategy(
        goal_id=goal.id,
        version=2,
        parent_version_id=v1.id,
        policy={"fit_floor": 0.65, "conversion_prior_factor": 0.40},
        created_at=now - timedelta(days=5),
        retired_at=None,
    )
    db_session.add_all([v1, v2])
    db_session.flush()

    # Rollback to v1
    v3 = rollback_strategy(db_session, goal, to_version=1, now=now)

    assert v3.version == 3
    assert v3.parent_version_id == v2.id
    assert v3.policy.get("fit_floor") == 0.50
    assert v2.retired_at == now
    assert v1.retired_at == now - timedelta(days=5)  # Historical row unmutated
