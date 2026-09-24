"""Unit tests for the observe module of Pilot Planner."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from pilot.db.models import ApplicationStage, Goal, GoalStatus
from pilot.planner.observe import _safe_div, observe
from pilot.planner.schemas import Observation


def test_safe_div():
    """Verify division safely returns 0.0 on zero-denominator and clamps to [0.0, 1.0]."""
    assert _safe_div(5, 0) == 0.0
    assert _safe_div(0, 0) == 0.0
    assert _safe_div(-1, 10) == 0.0
    assert _safe_div(5, 10) == 0.5
    assert _safe_div(15, 10) == 1.0


def test_observe_deterministic_and_pure():
    """Verify observe constructs an Observation correctly without LLM calls."""
    goal_id = uuid.uuid4()
    user_id = uuid.uuid4()
    created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    deadline = created_at + timedelta(days=90)
    now = created_at + timedelta(days=30)

    goal = Goal(
        id=goal_id,
        user_id=user_id,
        objective_text="Find Senior Distributed Systems Engineer role",
        status=GoalStatus.ACTIVE,
        constraints_json={},
        success_criteria={"min_offers": 1, "target_applications": 30},
        target_spec={},
        sub_goals=[
            {"id": "sg_1", "title": "Evidence", "metric_target": {"evidence_claims_verified": 10}},
            {"id": "sg_2", "title": "Sourcing", "metric_target": {"roles_identified": 50}},
            {
                "id": "sg_3",
                "title": "Applications",
                "metric_target": {"applications_submitted": 20},
            },
        ],
        deadline=deadline,
        created_at=created_at,
    )

    # Mock session
    mock_session = MagicMock()
    # scalars().first() for previous cycle: returns None
    # scalar() returns for counts:
    # 1st call: roles_sourced -> 50
    # 2nd call: roles_assessed -> 25
    # 3rd call: evidence_claims_count -> 10
    # 4th call: active_escalations_count -> 0
    # 5th call: avg_fit -> 0.85
    mock_session.scalar.side_effect = [50, 25, 10, 0, 0.85]

    # Application stages execute query: 10 APPLIED, 2 SCREENING, 1 OFFER
    app_stages = [
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.APPLIED,
        ApplicationStage.SCREENING,
        ApplicationStage.SCREENING,
        ApplicationStage.OFFER,
    ]
    exec_mock = MagicMock()
    exec_mock.scalars.return_value.all.return_value = app_stages
    exec_mock.scalars.return_value.first.return_value = None  # no previous cycle
    mock_session.execute.return_value = exec_mock

    obs = observe(mock_session, goal, now=now)

    assert isinstance(obs, Observation)
    assert obs.goal_id == goal_id
    assert obs.funnel_counts.roles_sourced == 50
    assert obs.funnel_counts.roles_assessed == 25
    assert obs.funnel_counts.applications == 11
    assert obs.funnel_counts.responses == 3  # 2 SCREENING + 1 OFFER
    assert obs.funnel_counts.interviews == 3
    assert obs.funnel_counts.offers == 1

    # Conversion rates
    assert obs.conversion_rates.sourcing_to_assessed == pytest.approx(0.5)
    assert obs.conversion_rates.assessed_to_applied == pytest.approx(11 / 25, abs=1e-4)
    assert obs.conversion_rates.applied_to_response == pytest.approx(3 / 11, abs=1e-4)

    # Timeline
    assert obs.timeline.elapsed_days == 30.0
    assert obs.timeline.remaining_days == 60.0
    assert obs.timeline.total_days == 90.0
    assert obs.timeline.pace_fraction == pytest.approx(30.0 / 90.0, abs=1e-4)

    # Sub-goals
    assert len(obs.sub_goal_progress) == 3
    assert obs.sub_goal_progress[0].actual_value == 10.0
    assert obs.sub_goal_progress[0].is_achieved is True
    assert obs.sub_goal_progress[1].actual_value == 50.0
    assert obs.sub_goal_progress[1].is_achieved is True
    assert obs.sub_goal_progress[2].actual_value == 11.0
    assert obs.sub_goal_progress[2].is_achieved is False


def test_observe_delta_calculation_with_prior_cycle():
    """Verify delta calculation between current state and previous cycle."""
    goal_id = uuid.uuid4()
    user_id = uuid.uuid4()
    created_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    now = created_at + timedelta(days=10)

    goal = Goal(
        id=goal_id,
        user_id=user_id,
        objective_text="Objective",
        status=GoalStatus.ACTIVE,
        constraints_json={},
        success_criteria={},
        target_spec={},
        sub_goals=[],
        deadline=created_at + timedelta(days=60),
        created_at=created_at,
    )

    mock_session = MagicMock()
    mock_session.scalar.side_effect = [20, 15, 8, 0, 0.75]

    # Applications
    exec_mock = MagicMock()
    exec_mock.scalars.return_value.all.return_value = [ApplicationStage.APPLIED] * 5

    # Prior cycle with cycle_number=1, counts: roles_sourced=10, applications=2
    prior_cycle = MagicMock()
    prior_cycle.cycle_number = 1
    prior_cycle.observation = {
        "funnel_counts": {
            "roles_sourced": 10,
            "roles_assessed": 8,
            "applications": 2,
            "responses": 0,
            "interviews": 0,
            "offers": 0,
        }
    }
    exec_mock.scalars.return_value.first.return_value = prior_cycle
    mock_session.execute.return_value = exec_mock

    obs = observe(mock_session, goal, now=now)

    assert obs.last_cycle_number == 1
    assert obs.delta_since_last_cycle["roles_sourced"] == 10  # 20 - 10
    assert obs.delta_since_last_cycle["roles_assessed"] == 7  # 15 - 8
    assert obs.delta_since_last_cycle["applications"] == 3  # 5 - 2
