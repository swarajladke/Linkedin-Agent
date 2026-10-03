"""Tests for Bayesian calibrated priors through the planner prediction seam and policy propagation."""

import uuid
from datetime import UTC, datetime, timedelta

from pilot.db.models import (
    Action,
    ActionOutcome,
    Company,
    Goal,
    Role,
    RoleAssessment,
    Strategy,
    User,
)
from pilot.planner.generate import generate_actions
from pilot.planner.predict import compute_prediction_prior, record_prediction
from pilot.planner.schemas import Diagnosis, DiagnosisCategory
from pilot.planner.score import score_actions


def test_predict_prior_bayesian_shrinkage_with_history(db_session):
    """Prediction seam shrinks probability using historical action outcomes."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    user = User(email="bayes@example.com", full_name="Bayes User")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Calibrated Prior Goal",
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
        policy={
            "default_conversion_prior": 0.70,
            "prior_strength": 10.0,
            "horizon_days": 21,
        },
    )
    db_session.add(strategy)
    db_session.flush()

    # 1. Zero history -> returns uncalibrated prior (0.70)
    action_1 = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Action 1",
    )
    prior_zero = compute_prediction_prior(action_1, session=db_session)
    assert prior_zero == 0.70

    # 2. Add 10 historical failures for this goal
    for i in range(10):
        past_action = Action(
            goal_id=goal.id,
            strategy_id=strategy.id,
            action_type="generate_application_package",
            target_type="role",
            target_id=uuid.uuid4(),
            reason=f"Past failure {i}",
            predicted_outcome="Will succeed",
            predicted_probability=0.70,
            executed_at=now - timedelta(days=10 + i),
        )
        db_session.add(past_action)
        db_session.flush()

        outcome = ActionOutcome(
            action=past_action,
            action_id=past_action.id,
            binary_success=False,
            brier_score=(0.70 - 0.0) ** 2,
            recorded_at=now - timedelta(days=5),
        )
        db_session.add(outcome)
    db_session.flush()

    # 3. New action should shrink from 0.70 down towards 0.0
    # Formula: (v0 * p0 + k) / (v0 + n) = (10 * 0.70 + 0) / (10 + 10) = 7.0 / 20 = 0.35
    action_shrunk = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Action after 10 failures",
    )
    prior_shrunk = compute_prediction_prior(action_shrunk, session=db_session)
    assert prior_shrunk == 0.35

    # 4. record_prediction populates calibrated probability and horizon_days from policy
    persisted = record_prediction(db_session, action_shrunk)
    assert persisted.predicted_probability == 0.35
    assert persisted.horizon_days == 21
    assert "within 21 days" in persisted.predicted_outcome


def test_planner_components_honor_strategy_policy(db_session):
    """Planner action generation and scoring honor parameters in strategy policy."""
    now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    user = User(email="policyman@example.com", full_name="Policy Man")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Policy Goal",
        constraints_json={},
        target_spec={},
        success_criteria={},
        sub_goals=[],
        deadline=now + timedelta(days=30),
    )
    db_session.add(goal)
    db_session.flush()

    # Strategy with a high fit_floor of 0.80
    custom_policy = {
        "fit_floor": 0.80,
        "elevated_fit_floor": 0.90,
        "conversion_prior_factor": 0.50,
        "urgency_multiplier_applications": 1.50,
    }
    strategy = Strategy(
        goal_id=goal.id,
        version=1,
        policy=custom_policy,
    )
    db_session.add(strategy)
    db_session.flush()

    company = Company(name="Policy Co")
    db_session.add(company)
    db_session.flush()

    role_low = Role(
        company_id=company.id,
        title="Junior Dev",
        location_type="remote",
        source="ashby",
        external_id="p_1",
    )
    role_high = Role(
        company_id=company.id,
        title="Lead Dev",
        location_type="remote",
        source="ashby",
        external_id="p_2",
    )
    db_session.add_all([role_low, role_high])
    db_session.flush()

    # Fit 0.70 (below 0.80 floor) and Fit 0.85 (above floor)
    assess_low = RoleAssessment(
        role_id=role_low.id,
        goal_id=goal.id,
        fit_score=0.70,
        fit_rationale="Decent fit",
        supporting_claim_ids=[],
        recommended_action="apply_now",
        assessor_version="1.0.0",
    )
    assess_high = RoleAssessment(
        role_id=role_high.id,
        goal_id=goal.id,
        fit_score=0.85,
        fit_rationale="Great fit",
        supporting_claim_ids=[],
        recommended_action="apply_now",
        assessor_version="1.0.0",
    )
    db_session.add_all([assess_low, assess_high])
    db_session.flush()

    diag = Diagnosis(category=DiagnosisCategory.STARVED, starved_stage="applications")

    # 1. Generation with custom policy admits only role_high
    candidates = generate_actions(db_session, goal, diag, policy=custom_policy)
    assert len(candidates) == 1
    assert candidates[0].role_id == role_high.id

    # 2. Scoring with custom policy uses custom conversion_prior_factor (0.50) and urgency (1.50)
    scored = score_actions(candidates, diag, policy=custom_policy)
    assert len(scored) == 1
    # predicted_prob = fit_score (0.85) * factor (0.50) = 0.425
    assert scored[0].predicted_probability == 0.425
    assert scored[0].urgency == 1.50
