"""Mathematical integration tests asserting sub-goal ordering, monotonic funnel volumes, and determinism."""

import math
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Goal, User
from pilot.extraction.llm import StructuredLLMClient
from pilot.goals.compiler import GoalCompiler
from pilot.goals.errors import InfeasibleGoalError
from pilot.goals.repository import persist_compiled_goal
from pilot.goals.schemas import CompiledGoalDraft, FunnelAssumptions
from pilot.schemas.goal import TargetSpec

pytestmark = pytest.mark.integration

T = TypeVar("T", bound=BaseModel)

PRIOR_SETS = {
    "optimistic": FunnelAssumptions(
        application_to_response=0.20,
        response_to_interview=0.40,
        interview_to_offer=0.30,
        sourcing_to_application=0.60,
    ),
    "pessimistic": FunnelAssumptions(
        application_to_response=0.08,
        response_to_interview=0.15,
        interview_to_offer=0.10,
        sourcing_to_application=0.40,
    ),
}


class DeterministicGoalLLM(StructuredLLMClient):
    """Fake LLM returning a pre-specified compiled goal draft."""

    def __init__(self, draft: CompiledGoalDraft) -> None:
        self.draft = draft

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        return self.draft  # type: ignore


@pytest.mark.parametrize("deadline_days", [14, 45, 90, 180])
@pytest.mark.parametrize("min_offers", [1, 3])
@pytest.mark.parametrize("priors_name", ["optimistic", "pessimistic"])
def test_parametrized_goal_timeline_and_funnel_invariants(
    session: Session,
    seeded_user: User,
    deadline_days: int,
    min_offers: int,
    priors_name: str,
):
    """Invariant: Sub-goal deadlines must be strictly increasing in dependency order within (now, deadline), funnel volumes must be strictly monotonic up the funnel, and persisted criteria must be mathematically self-consistent with funnel priors."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=deadline_days)
    priors = PRIOR_SETS[priors_name]

    draft = CompiledGoalDraft(
        target_spec=TargetSpec(
            must_have=["Staff AI Engineer", "Remote"],
            nice_to_have=["Fintech", "Series B"],
            unstated_but_real=["Production systems design"],
        ),
        success_criteria={
            "min_offers": min_offers,
            "min_salary": 4500000,
        },
        funnel_assumptions=priors,
    )

    llm = DeterministicGoalLLM(draft)
    compiler = GoalCompiler(llm=llm, now=fixed_now)

    compiled_goal = compiler.compile(
        user_id=seeded_user.id,
        objective_text="Secure a Staff AI Engineer offer",
        constraints={},
        deadline=deadline,
    )

    persisted_goal, _ = persist_compiled_goal(session, compiled_goal)
    session.commit()

    db_goal = session.get(Goal, persisted_goal.id)
    assert db_goal is not None

    # Invariant 1: sub-goal deadlines strictly increasing in depends_on order
    sub_goals = db_goal.sub_goals
    assert len(sub_goals) == 4

    parsed_deadlines: list[datetime] = []
    for sg in sub_goals:
        raw_dl = sg["deadline"]
        dt = datetime.fromisoformat(raw_dl) if isinstance(raw_dl, str) else raw_dl
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        parsed_deadlines.append(dt)

    d1, d2, d3, d4 = parsed_deadlines

    # Strictly increasing and properly bounded: now < d1 < d2 < d3 < d4 < deadline
    assert fixed_now < d1 < d2 < d3 < d4 < deadline
    assert sub_goals[0]["depends_on"] == []
    assert sub_goals[1]["depends_on"] == ["sg_1"]
    assert sub_goals[2]["depends_on"] == ["sg_2"]
    assert sub_goals[3]["depends_on"] == ["sg_3"]

    # Invariant 2: back-solved volumes are monotonic up the funnel:
    # roles_to_source >= applications_needed >= responses_needed >= interviews_needed >= min_offers
    min_offers_val = db_goal.success_criteria["min_offers"]
    interviews_needed = sub_goals[3]["metric_target"]["interviews_completed"]
    responses_needed = math.ceil(interviews_needed / priors.response_to_interview)
    applications_needed = db_goal.success_criteria["target_applications"]
    roles_to_source = sub_goals[1]["metric_target"]["roles_identified"]

    assert roles_to_source >= applications_needed
    assert applications_needed >= responses_needed
    assert responses_needed >= interviews_needed
    assert interviews_needed >= min_offers_val

    # Invariant 3: recomputing the funnel from persisted criteria reproduces identical volumes
    expected_interviews = math.ceil(min_offers_val / priors.interview_to_offer)
    expected_responses = math.ceil(expected_interviews / priors.response_to_interview)
    expected_applications = math.ceil(expected_responses / priors.application_to_response)
    expected_roles = math.ceil(expected_applications / (priors.sourcing_to_application or 0.5))

    assert interviews_needed == expected_interviews
    assert responses_needed == expected_responses
    assert applications_needed == expected_applications
    assert roles_to_source == expected_roles
    assert sub_goals[2]["metric_target"]["applications_submitted"] == expected_applications


def test_infeasible_goal_rate_constraint_prevents_persistence(
    session: Session,
    seeded_user: User,
):
    """Invariant: When a rate-limiting constraint renders the back-solved funnel mathematically impossible within the available days, an InfeasibleGoalError must be raised and zero goal records persisted."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=14)

    draft = CompiledGoalDraft(
        target_spec=TargetSpec(
            must_have=["Principal Engineer"],
            nice_to_have=[],
            unstated_but_real=[],
        ),
        success_criteria={"min_offers": 3},
        funnel_assumptions=FunnelAssumptions(
            application_to_response=0.05,
            response_to_interview=0.20,
            interview_to_offer=0.10,
            sourcing_to_application=0.50,
        ),
    )

    # 3 offers / 0.10 = 30 interviews; 30 / 0.20 = 150 responses; 150 / 0.05 = 3000 applications.
    # Over 14 days, 3000 apps requires ~214 apps/day.
    # Imposing a max of 2 apps/day renders this completely infeasible.
    llm = DeterministicGoalLLM(draft)
    compiler = GoalCompiler(llm=llm, now=fixed_now)

    with pytest.raises(InfeasibleGoalError) as exc_info:
        compiler.compile(
            user_id=seeded_user.id,
            objective_text="Land 3 Principal Engineer offers in two weeks",
            constraints={"max_applications_per_day": 2},
            deadline=deadline,
        )

    assert "exceeding constraint of 2.0/day" in str(exc_info.value)

    # Verify nothing was persisted to database
    goals = session.scalars(select(Goal).where(Goal.user_id == seeded_user.id)).all()
    assert len(goals) == 0


def test_goal_compiler_deterministic_compilation(seeded_user: User):
    """Invariant: The goal compiler must be purely deterministic, producing identical structured goal specifications, metric targets, and timelines given identical inputs and reference time."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    deadline = fixed_now + timedelta(days=60)

    draft = CompiledGoalDraft(
        target_spec=TargetSpec(
            must_have=["AI Research Engineer", "Bangalore"],
            nice_to_have=["High-growth startup"],
            unstated_but_real=["PyTorch", "CUDA profiling"],
        ),
        success_criteria={"min_offers": 1},
        funnel_assumptions=FunnelAssumptions(
            application_to_response=0.10,
            response_to_interview=0.25,
            interview_to_offer=0.20,
            sourcing_to_application=0.50,
        ),
    )

    llm = DeterministicGoalLLM(draft)
    compiler = GoalCompiler(llm=llm, now=fixed_now)

    compiled_1 = compiler.compile(
        user_id=seeded_user.id,
        objective_text="Land AI Research Engineer role",
        constraints={"max_applications_per_day": 10},
        deadline=deadline,
    )

    compiled_2 = compiler.compile(
        user_id=seeded_user.id,
        objective_text="Land AI Research Engineer role",
        constraints={"max_applications_per_day": 10},
        deadline=deadline,
    )

    # Both compilation passes must produce identical dumps
    assert compiled_1.model_dump() == compiled_2.model_dump()
