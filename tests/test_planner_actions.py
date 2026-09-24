"""Unit tests for action generation, scoring, prediction, and execution."""

import random
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from pilot.db.models import (
    Action,
    Company,
    Escalation,
    EvidenceClaim,
    Goal,
    Role,
    RoleAssessment,
)
from pilot.planner.execute import execute
from pilot.planner.predict import record_prediction
from pilot.planner.schemas import (
    CandidateAction,
    Diagnosis,
    DiagnosisCategory,
)
from pilot.planner.score import score_actions, select_actions


def _make_candidate(
    role_title: str, fit_score: float, role_id: uuid.UUID | None = None
) -> CandidateAction:
    rid = role_id or uuid.uuid4()
    return CandidateAction(
        action_type="generate_application_package",
        target_type="role",
        target_id=rid,
        role_id=rid,
        role_title=role_title,
        company_name="Acme Corp",
        fit_score=fit_score,
        supporting_claim_ids=[str(uuid.uuid4())],
        skill_gaps=[],
        reason=f"Fit score {fit_score} for {role_title}",
    )


def test_score_actions_determinism_under_shuffle():
    """Shuffled inputs must yield identical scores and identical ordering."""
    candidates = [
        _make_candidate("Staff Engineer", 0.95),
        _make_candidate("Senior Engineer A", 0.85),
        _make_candidate("Senior Engineer B", 0.85),  # same fit score -> tie break on role_id
        _make_candidate("Backend Engineer", 0.70),
    ]

    diag = Diagnosis(
        category=DiagnosisCategory.STARVED,
        starved_stage="applications",
    )

    scored_original = score_actions(candidates, diag)

    # Shuffle 10 times and verify output is identical every time
    for seed in range(10):
        shuffled = list(candidates)
        random.Random(seed).shuffle(shuffled)
        scored_shuffled = score_actions(shuffled, diag)

        assert len(scored_shuffled) == len(scored_original)
        for s_orig, s_shuf in zip(scored_original, scored_shuffled, strict=True):
            assert s_orig.candidate.role_id == s_shuf.candidate.role_id
            assert s_orig.score == s_shuf.score
            assert s_orig.predicted_probability == s_shuf.predicted_probability


def test_select_actions_budget_and_rate_capping():
    """Selection strictly obeys budget limit and max_actions per cycle."""
    candidates = [_make_candidate(f"Engineer {i}", 0.90 - (i * 0.05)) for i in range(10)]
    diag = Diagnosis(category=DiagnosisCategory.ON_PACE)
    scored = score_actions(candidates, diag)

    # 1. Cap to max 3 actions
    selected_capped = select_actions(scored, max_actions=3)
    assert len(selected_capped) == 3
    # Most urgent/highest scored chosen
    assert selected_capped[0].candidate.role_title == "Engineer 0"
    assert selected_capped[1].candidate.role_title == "Engineer 1"
    assert selected_capped[2].candidate.role_title == "Engineer 2"

    # 2. Budget constraint
    # Set positive costs on actions
    for s in scored:
        s.cost = Decimal("10.00")

    selected_budgeted = select_actions(scored, budget=Decimal("25.00"))
    assert len(selected_budgeted) == 2  # 10 + 10 = 20 <= 25, 3rd would be 30 > 25


def test_execute_without_persisted_prediction_raises():
    """execute() structurally refuses to execute an action without a persisted prediction."""
    mock_session = MagicMock()
    now = datetime.now(UTC)

    # 1. Action with no ID (unpersisted)
    action_unpersisted = Action(
        goal_id=uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Test",
    )
    with pytest.raises(AssertionError, match="Cannot execute unpersisted action"):
        execute(mock_session, action_unpersisted, now=now)

    # 2. Action with ID but missing predicted_outcome
    action_no_prediction = Action(
        id=uuid.uuid4(),
        goal_id=uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Test",
        predicted_outcome="",
        predicted_probability=0.7,
    )
    with pytest.raises(AssertionError, match="no persisted prediction"):
        execute(mock_session, action_no_prediction, now=now)

    # 3. Action with ID but missing predicted_probability
    action_no_prob = Action(
        id=uuid.uuid4(),
        goal_id=uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Test",
        predicted_outcome="Will get interview",
        predicted_probability=None,
    )
    with pytest.raises(AssertionError, match="no persisted prediction"):
        execute(mock_session, action_no_prob, now=now)


def test_record_prediction_persists_prior_and_sentence():
    """record_prediction properly flushes action with prediction before execution."""
    mock_session = MagicMock()
    action = Action(
        goal_id=uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        action_type="generate_application_package",
        target_type="role",
        target_id=uuid.uuid4(),
        reason="Test reason",
    )

    persisted = record_prediction(mock_session, action, prior=0.75)

    assert persisted.predicted_probability == 0.75
    assert len(persisted.predicted_outcome) > 10
    mock_session.add.assert_called_once_with(action)
    mock_session.flush.assert_called_once()


def test_execute_creates_grounded_draft_and_escalation():
    """execute() builds draft from supporting claims and logs an Escalation."""
    mock_session = MagicMock()
    now = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)

    role_id = uuid.uuid4()
    company_id = uuid.uuid4()
    goal_id = uuid.uuid4()
    user_id = uuid.uuid4()
    claim_id = uuid.uuid4()

    action = Action(
        id=uuid.uuid4(),
        goal_id=goal_id,
        strategy_id=uuid.uuid4(),
        action_type="generate_application_package",
        target_type="role",
        target_id=role_id,
        reason="Grounded package generation",
        predicted_outcome="Advances to technical round",
        predicted_probability=0.80,
    )

    role = Role(id=role_id, title="Principal Infrastructure Engineer", company_id=company_id)
    company = Company(id=company_id, name="CloudScale Systems")
    goal = Goal(id=goal_id, user_id=user_id, objective_text="Find job")
    assessment = RoleAssessment(
        role_id=role_id,
        goal_id=goal_id,
        fit_score=0.92,
        fit_rationale="Great fit",
        supporting_claim_ids=[str(claim_id)],
        recommended_action="apply_now",
        assessor_version="1.0.0",
    )
    claim = EvidenceClaim(
        id=claim_id,
        entity_type="user",
        entity_id=user_id,
        claim="Built high-throughput Raft consensus engine in Rust",
        source="github",
        source_excerpt="Built high-throughput Raft consensus engine in Rust scaling to 100k ops/sec",
        source_url="https://github.com/candidate/raft-rs",
        content_hash="a" * 64,
        confidence=0.95,
    )

    def mock_get(model, pk):
        if model == Role:
            return role
        if model == Company:
            return company
        if model == Goal:
            return goal
        return None

    mock_session.get.side_effect = mock_get

    # execute query returns
    exec_assessment = MagicMock()
    exec_assessment.scalars.return_value.first.return_value = assessment

    exec_claims = MagicMock()
    exec_claims.scalars.return_value.all.return_value = [claim]

    mock_session.execute.side_effect = [exec_assessment, exec_claims]

    result = execute(mock_session, action, now=now)

    assert result.action_id == action.id
    assert result.executed_at == now
    assert action.executed_at == now
    assert action.actual_outcome is not None
    assert str(claim_id) in result.draft_package.supporting_claim_ids

    # Verified escalation added
    mock_session.add.assert_called()
    added_obj = mock_session.add.call_args[0][0]
    assert isinstance(added_obj, Escalation)
    assert added_obj.action_id == action.id
    assert added_obj.resolved is False
