"""Unit tests for calibration math, Murphy decomposition, and Bayesian shrinkage."""

from datetime import UTC, datetime

import pytest

from pilot.db.models import Action, ActionOutcome, Goal, Strategy, User
from pilot.learning.calibration import calibrated_probability, compute_calibration


def test_calibration_hand_computed_fixture(db_session):
    """
    Verify calibration metrics against analytical hand-computed values:
    4 outcomes:
    p=0.25, o=0
    p=0.25, o=1
    p=0.75, o=1
    p=0.75, o=1

    N=4, base_rate=0.75
    Brier = 0.1875
    ECE = 0.25
    REL = 0.0625, RES = 0.0625, UNC = 0.1875
    Murphy BS = REL - RES + UNC = 0.1875
    """
    user = User(email="calib@example.com", full_name="Calib User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Test Calibration Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=datetime.now(UTC),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    # Create the 4 actions and outcomes
    data = [(0.25, False), (0.25, True), (0.75, True), (0.75, True)]
    for p, o in data:
        action = Action(
            goal_id=goal.id,
            strategy_id=strategy.id,
            action_type="generate_application_package",
            target_type="role",
            reason="Calibration test",
            predicted_outcome="Interview",
            predicted_probability=p,
            executed_at=datetime.now(UTC),
        )
        db_session.add(action)
        db_session.flush()

        outcome = ActionOutcome(
            action_id=action.id,
            binary_success=o,
            brier_score=(p - (1.0 if o else 0.0)) ** 2,
            recorded_at=datetime.now(UTC),
        )
        db_session.add(outcome)

    db_session.flush()

    report = compute_calibration(db_session, strategy.id)

    assert report.sample_size == 4
    assert pytest.approx(report.base_rate, rel=1e-4) == 0.75
    assert pytest.approx(report.mean_brier_score, rel=1e-4) == 0.1875
    assert pytest.approx(report.ece, rel=1e-4) == 0.25

    murphy = report.murphy_decomposition
    assert pytest.approx(murphy.reliability, rel=1e-4) == 0.0625
    assert pytest.approx(murphy.resolution, rel=1e-4) == 0.0625
    assert pytest.approx(murphy.uncertainty, rel=1e-4) == 0.1875
    # Murphy decomposition identity: REL - RES + UNC == BS
    assert (
        pytest.approx(murphy.reliability - murphy.resolution + murphy.uncertainty, rel=1e-4)
        == murphy.brier_score
    )
    assert pytest.approx(murphy.brier_score, rel=1e-4) == report.mean_brier_score


def test_perfectly_calibrated_synthetic_predictor(db_session):
    """A predictor whose predicted probabilities match observed rates has ECE approx 0."""
    user = User(email="perfect@example.com", full_name="Perfect Predictor")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Perfect Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=datetime.now(UTC),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    # 10 buckets, each with 10 outcomes matching the bucket prediction
    for i in range(10):
        p = round(i * 0.1 + 0.05, 2)
        success_count = int(round(p * 20))
        total_in_bucket = 20
        for j in range(total_in_bucket):
            success = j < success_count
            action = Action(
                goal_id=goal.id,
                strategy_id=strategy.id,
                action_type="generate_application_package",
                target_type="role",
                reason="Perfect calibration",
                predicted_outcome="Interview",
                predicted_probability=p,
                executed_at=datetime.now(UTC),
            )
            db_session.add(action)
            db_session.flush()

            outcome = ActionOutcome(
                action_id=action.id,
                binary_success=success,
                brier_score=(p - (1.0 if success else 0.0)) ** 2,
                recorded_at=datetime.now(UTC),
            )
            db_session.add(outcome)

    db_session.flush()

    report = compute_calibration(db_session, strategy.id)
    assert report.sample_size == 200
    # ECE should be very close to 0 (< 0.05)
    assert report.ece < 0.05
    # Murphy identity holds
    m = report.murphy_decomposition
    assert pytest.approx(m.reliability - m.resolution + m.uncertainty, abs=1e-4) == m.brier_score


def test_constant_overconfident_predictor_negative_skill(db_session):
    """A constant 0.9 predictor on a 10% base rate gets large ECE and negative skill."""
    user = User(email="overconf@example.com", full_name="Overconfident")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Overconfident Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=datetime.now(UTC),
    )
    db_session.add(goal)
    db_session.flush()

    strategy = Strategy(goal_id=goal.id, version=1, policy={})
    db_session.add(strategy)
    db_session.flush()

    # 100 predictions of 0.9, but only 10 successes (10% base rate)
    for j in range(100):
        success = j < 10
        action = Action(
            goal_id=goal.id,
            strategy_id=strategy.id,
            action_type="generate_application_package",
            target_type="role",
            reason="Overconfident",
            predicted_outcome="Interview",
            predicted_probability=0.9,
            executed_at=datetime.now(UTC),
        )
        db_session.add(action)
        db_session.flush()

        outcome = ActionOutcome(
            action_id=action.id,
            binary_success=success,
            brier_score=(0.9 - (1.0 if success else 0.0)) ** 2,
            recorded_at=datetime.now(UTC),
        )
        db_session.add(outcome)

    db_session.flush()

    report = compute_calibration(db_session, strategy.id)
    assert report.sample_size == 100
    assert pytest.approx(report.base_rate, rel=1e-3) == 0.10
    # ECE is large (|0.9 - 0.1| = 0.8)
    assert pytest.approx(report.ece, rel=1e-3) == 0.80
    # Negative skill score against base rate
    assert report.brier_skill_score < 0.0
    assert not report.beats_base_rate


def test_bayesian_shrinkage_properties():
    """Verify Bayesian shrinkage estimator properties."""
    prior = 0.40

    # 1. Zero data returns prior
    assert calibrated_probability(prior, []) == prior
    assert calibrated_probability(prior, None) == prior

    # 2. Monotonic in observed successes (holding n fixed)
    p_1_success = calibrated_probability(prior, [True, False, False, False, False])
    p_3_success = calibrated_probability(prior, [True, True, True, False, False])
    p_5_success = calibrated_probability(prior, [True, True, True, True, True])

    assert p_1_success < p_3_success < p_5_success

    # 3. As n grows, converges to observed rate (e.g. 100% successes -> converges to 0.99 clamp)
    large_successes = [True] * 1000
    assert calibrated_probability(prior, large_successes) >= 0.98

    # 4. As n grows with zero successes, converges to 0.01 clamp
    large_failures = [False] * 1000
    assert calibrated_probability(prior, large_failures) <= 0.02


def test_calibration_idempotent_persistence(db_session):
    """Recomputing calibration over the same outcomes updates the existing record idempotently."""
    user = User(email="idem@example.com", full_name="Idem User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Idempotence Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=datetime.now(UTC),
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
        reason="Idem test",
        predicted_outcome="Interview",
        predicted_probability=0.6,
        executed_at=datetime.now(UTC),
    )
    db_session.add(action)
    db_session.flush()

    outcome = ActionOutcome(
        action_id=action.id,
        binary_success=True,
        brier_score=(0.6 - 1.0) ** 2,
        recorded_at=datetime.now(UTC),
    )
    db_session.add(outcome)
    db_session.flush()

    # First call
    rep1 = compute_calibration(db_session, strategy.id)
    # Second call
    rep2 = compute_calibration(db_session, strategy.id)

    assert rep1.sample_size == rep2.sample_size == 1
    assert rep1.mean_brier_score == rep2.mean_brier_score
    assert len(strategy.calibrations) == 1
