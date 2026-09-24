"""Unit and integration tests for GoalCompiler, back-solving timelines, and persistence."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from pilot.db.models import Goal, GoalStatus, Strategy, StrategyNote, User
from pilot.goals import (
    CompiledGoalDraft,
    FunnelAssumptions,
    GoalCompiler,
    InfeasibleGoalError,
    persist_compiled_goal,
)
from pilot.schemas.goal import TargetSpec

T = TypeVar("T", bound=BaseModel)


class FakeGoalLLM:
    """Fake LLM client providing deterministic structured goal drafts."""

    def __init__(self, draft: CompiledGoalDraft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.calls.append((system_prompt, user_prompt))
        return self.draft  # type: ignore


@pytest.fixture
def sample_draft() -> CompiledGoalDraft:
    """Provide a standard draft with realistic priors and mixed success criteria."""
    return CompiledGoalDraft(
        target_spec=TargetSpec(
            must_have=["Applied AI Engineer", "Remote or Bangalore", "Min INR 35L"],
            nice_to_have=["Series A/B startup", "Python & PyTorch stack"],
            unstated_but_real=[
                "Can pass live DSA coding screen",
                "Verifiable LLM evaluation project",
                "IST overlap > 4 hours",
            ],
        ),
        success_criteria={
            "min_offers": 1,
            "min_salary": 3500000,
            "min_response_rate": 0.15,
            "preferred_city": "Bangalore",  # Non-numeric string to strip
            "remote_only": True,  # Boolean to strip
            "invalid_str": "not_a_number",  # Non-numeric string to strip
        },
        funnel_assumptions=FunnelAssumptions(
            application_to_response=0.10,
            response_to_interview=0.25,
            interview_to_offer=0.20,
            sourcing_to_application=0.50,
        ),
    )


def test_target_spec_classification_shape(sample_draft):
    """Test that compiled goal contains all three target_spec buckets properly classified."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=60)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    user_id = uuid.uuid4()
    compiled = compiler.compile(
        user_id=user_id,
        objective_text="Land Applied AI Engineer role by Dec 1",
        constraints={"max_applications_per_day": 10},
        deadline=deadline,
    )

    assert len(compiled.target_spec.must_have) == 3
    assert "Applied AI Engineer" in compiled.target_spec.must_have
    assert len(compiled.target_spec.nice_to_have) == 2
    assert len(compiled.target_spec.unstated_but_real) == 3
    assert "Can pass live DSA coding screen" in compiled.target_spec.unstated_but_real


def test_non_numeric_success_criteria_are_stripped(sample_draft):
    """Test that string, boolean, and non-numeric fields in success_criteria are purged."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=60)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    compiled = compiler.compile(
        user_id=uuid.uuid4(),
        objective_text="Land Applied AI Engineer role",
        constraints={},
        deadline=deadline,
    )

    criteria = compiled.success_criteria
    assert "preferred_city" not in criteria
    assert "remote_only" not in criteria
    assert "invalid_str" not in criteria

    # Numeric keys must remain intact
    assert criteria["min_offers"] == 1
    assert criteria["min_salary"] == 3500000
    assert criteria["min_response_rate"] == 0.15
    for _k, v in criteria.items():
        assert isinstance(v, int | float) and not isinstance(v, bool)


def test_back_solved_volumes_mathematically_correct(sample_draft):
    """Test funnel volume calculations against known priors.

    min_offers = 1
    interview_to_offer = 0.20 -> ceil(1 / 0.20) = 5 interviews
    response_to_interview = 0.25 -> ceil(5 / 0.25) = 20 responses
    application_to_response = 0.10 -> ceil(20 / 0.10) = 200 applications
    sourcing_to_application = 0.50 -> ceil(200 / 0.50) = 400 roles
    """
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=60)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    compiled = compiler.compile(
        user_id=uuid.uuid4(),
        objective_text="Land Applied AI Engineer role",
        constraints={},
        deadline=deadline,
    )

    sub_goals_by_id = {sg.id: sg for sg in compiled.sub_goals}

    assert sub_goals_by_id["sg_2"].metric_target["roles_identified"] == 400
    assert sub_goals_by_id["sg_3"].metric_target["applications_submitted"] == 200
    assert sub_goals_by_id["sg_4"].metric_target["interviews_completed"] == 5
    assert sub_goals_by_id["sg_4"].metric_target["offers_received"] == 1
    assert compiled.success_criteria["target_applications"] == 200


def test_sub_goal_deadlines_strictly_ordered_and_precede_goal_deadline(sample_draft):
    """Test that all sub-goals strictly lie inside (now, goal.deadline) and increase in order."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=90)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    compiled = compiler.compile(
        user_id=uuid.uuid4(),
        objective_text="Land Applied AI Engineer role",
        constraints={},
        deadline=deadline,
    )

    sub_goals = compiled.sub_goals
    assert len(sub_goals) == 4

    previous_deadline = fixed_now
    for sg in sub_goals:
        assert sg.deadline > fixed_now
        assert sg.deadline < deadline
        assert sg.deadline > previous_deadline
        previous_deadline = sg.deadline


def test_infeasible_goal_error_for_past_deadline(sample_draft):
    """Test that a deadline in the past raises InfeasibleGoalError."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    past_deadline = fixed_now - timedelta(days=1)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    with pytest.raises(InfeasibleGoalError, match="in the past"):
        compiler.compile(
            user_id=uuid.uuid4(),
            objective_text="Impossible past goal",
            constraints={},
            deadline=past_deadline,
        )


def test_infeasible_goal_error_for_over_constrained_rate_cap(sample_draft):
    """Test that impossible application volumes for the given window raise InfeasibleGoalError.

    Applications needed = 200.
    Window = 10 days.
    Max rate = 5 apps/day (can do at most 50 apps).
    Must raise InfeasibleGoalError without silently clamping.
    """
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    short_deadline = fixed_now + timedelta(days=10)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    with pytest.raises(InfeasibleGoalError, match="exceeding constraint"):
        compiler.compile(
            user_id=uuid.uuid4(),
            objective_text="Rush goal with tight rate limit",
            constraints={"max_applications_per_day": 5},
            deadline=short_deadline,
        )


def test_persist_compiled_goal_db_integration(db_session, sample_draft):
    """Integration test verifying persistence of Goal, Strategy v1, StrategyNote, and single ACTIVE constraint."""
    # 1. Create a user
    user = User(
        email=f"candidate_{uuid.uuid4().hex[:8]}@example.com",
        full_name="Ada Lovelace",
    )
    db_session.add(user)
    db_session.flush()

    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=60)
    fake_llm = FakeGoalLLM(sample_draft)
    compiler = GoalCompiler(llm=fake_llm, now=fixed_now)

    compiled_1 = compiler.compile(
        user_id=user.id,
        objective_text="Land Applied AI Engineer role, remote or Bangalore",
        constraints={"max_applications_per_day": 10},
        deadline=deadline,
    )

    # 2. Persist initial goal
    db_goal_1, db_strategy_1 = persist_compiled_goal(db_session, compiled_1)

    assert db_goal_1.id is not None
    assert db_goal_1.status == GoalStatus.ACTIVE
    assert db_strategy_1.id is not None
    assert db_strategy_1.version == 1
    assert db_strategy_1.parent_version_id is None

    # Check baseline StrategyNote
    notes = db_session.scalars(
        select(StrategyNote).where(StrategyNote.goal_id == db_goal_1.id)
    ).all()
    assert len(notes) == 1
    assert "Baseline Strategy v1" in notes[0].hypothesis
    assert notes[0].strategy_id == db_strategy_1.id

    # Check Strategy query
    strategies = db_session.scalars(select(Strategy).where(Strategy.goal_id == db_goal_1.id)).all()
    assert len(strategies) == 1
    assert strategies[0].version == 1

    # 3. Compile and persist a second goal for the same user — must pause previous active goal
    compiled_2 = compiler.compile(
        user_id=user.id,
        objective_text="Pivot: Land ML Platform Engineer role",
        constraints={"max_applications_per_day": 10},
        deadline=deadline + timedelta(days=30),
    )
    db_goal_2, db_strategy_2 = persist_compiled_goal(db_session, compiled_2)

    db_session.refresh(db_goal_1)
    assert db_goal_1.status == GoalStatus.PAUSED
    assert db_goal_2.status == GoalStatus.ACTIVE
    assert db_strategy_2.version == 1

    # Assert exactly 1 active goal exists for user
    active_goals = db_session.scalars(
        select(Goal).where(Goal.user_id == user.id, Goal.status == GoalStatus.ACTIVE)
    ).all()
    assert len(active_goals) == 1
    assert active_goals[0].id == db_goal_2.id
