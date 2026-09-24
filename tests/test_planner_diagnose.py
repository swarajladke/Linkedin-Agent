"""Unit tests for the deterministic funnel pace diagnoser."""

import uuid
from datetime import UTC, datetime, timedelta

from pilot.db.models import Goal, GoalStatus
from pilot.extraction.llm import StructuredLLMClient
from pilot.planner.diagnose import diagnose
from pilot.planner.schemas import (
    ConversionRates,
    DiagnosisCategory,
    FunnelStageCounts,
    Observation,
    RootCauseHypothesis,
    RootCauseKind,
    TimelineProgress,
)


class ScriptedLLM(StructuredLLMClient):
    """Fake structured LLM client for tests."""

    def __init__(self, canned_response):
        self.canned_response = canned_response
        self.calls = []

    def complete_structured(self, system_prompt: str, user_prompt: str, schema):
        self.calls.append((system_prompt, user_prompt))
        return self.canned_response


def _make_goal(
    *,
    status: GoalStatus = GoalStatus.ACTIVE,
    constraints: dict | None = None,
    success_criteria: dict | None = None,
    sub_goals: list | None = None,
) -> Goal:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    goal = Goal(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        objective_text="Secure a Senior Distributed Systems Engineer role in NYC or Remote",
        status=status,
        constraints_json=constraints or {"max_applications_per_day": 5},
        success_criteria=success_criteria or {"min_offers": 1, "target_applications": 32},
        target_spec={},
        sub_goals=sub_goals
        or [
            {"id": "sg_1", "title": "Evidence", "metric_target": {"evidence_claims_verified": 10}},
            {"id": "sg_2", "title": "Sourcing", "metric_target": {"roles_identified": 64}},
            {
                "id": "sg_3",
                "title": "Applications",
                "metric_target": {"applications_submitted": 32},
            },
            {
                "id": "sg_4",
                "title": "Interviews",
                "metric_target": {"interviews_completed": 4, "offers_received": 1},
            },
        ],
        deadline=now + timedelta(days=90),
        created_at=now,
    )
    return goal


def test_worked_example_32_roles_8_apps_0_responses():
    """
    Build spec worked example:
    32 relevant roles / 8 applications / 0 responses must yield:
    sourcing HEALTHY, applications HEALTHY, responses STARVED — never 'find more jobs'.
    """
    goal = _make_goal()
    obs = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at + timedelta(days=22, hours=12),  # ~25% pace
        funnel_counts=FunnelStageCounts(
            roles_sourced=32,
            roles_assessed=32,
            applications=8,
            responses=0,
            interviews=0,
            offers=0,
        ),
        conversion_rates=ConversionRates(
            sourcing_to_assessed=1.0,
            assessed_to_applied=0.25,
            applied_to_response=0.0,
            response_to_interview=0.0,
            interview_to_offer=0.0,
        ),
        timeline=TimelineProgress(
            elapsed_days=22.5,
            remaining_days=67.5,
            total_days=90.0,
            pace_fraction=0.25,
        ),
        evidence_claims_count=12,
        active_escalations_count=0,
    )

    diag = diagnose(goal, obs)

    assert diag.category == DiagnosisCategory.STARVED
    assert diag.starved_stage == "responses"
    assert diag.stage_health["sourcing"] == "HEALTHY"
    assert diag.stage_health["applications"] == "HEALTHY"
    assert diag.stage_health["responses"] == "STARVED"
    # Never diagnose sourcing starvation when sourcing is healthy
    assert diag.starved_stage != "sourcing"
    assert len(diag.hypotheses) > 0


def test_diagnosis_branch_blocked():
    """Verify all blocked conditions."""
    # 1. Paused goal
    goal_paused = _make_goal(status=GoalStatus.PAUSED)
    obs = Observation(
        goal_id=goal_paused.id,
        observed_at=goal_paused.created_at,
        funnel_counts=FunnelStageCounts(roles_sourced=10),
        conversion_rates=ConversionRates(),
        timeline=TimelineProgress(),
        evidence_claims_count=10,
    )
    diag = diagnose(goal_paused, obs)
    assert diag.category == DiagnosisCategory.BLOCKED
    assert "paused" in diag.blocked_reason.lower()

    # 2. Active escalations
    goal = _make_goal()
    obs_escalation = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at,
        funnel_counts=FunnelStageCounts(),
        conversion_rates=ConversionRates(),
        timeline=TimelineProgress(),
        evidence_claims_count=10,
        active_escalations_count=2,
    )
    diag_esc = diagnose(goal, obs_escalation)
    assert diag_esc.category == DiagnosisCategory.BLOCKED
    assert "escalation" in diag_esc.blocked_reason.lower()

    # 3. No verified evidence claims
    obs_no_claims = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at,
        funnel_counts=FunnelStageCounts(),
        conversion_rates=ConversionRates(),
        timeline=TimelineProgress(),
        evidence_claims_count=0,
    )
    diag_no_claims = diagnose(goal, obs_no_claims)
    assert diag_no_claims.category == DiagnosisCategory.BLOCKED
    assert "evidence claims" in diag_no_claims.blocked_reason.lower()


def test_diagnosis_branch_constraint_conflict():
    """Verify constraint conflict when required rate exceeds daily cap."""
    # Goal needs 32 applications, 2 days remaining, but cap is 2 per day -> 16/day required > 2/day
    goal = _make_goal(constraints={"max_applications_per_day": 2})
    obs = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at + timedelta(days=88),
        funnel_counts=FunnelStageCounts(applications=0),
        conversion_rates=ConversionRates(),
        timeline=TimelineProgress(
            elapsed_days=88.0,
            remaining_days=2.0,
            total_days=90.0,
            pace_fraction=0.98,
        ),
        evidence_claims_count=10,
    )
    diag = diagnose(goal, obs)
    assert diag.category == DiagnosisCategory.CONSTRAINT_CONFLICT
    assert "exceeds" in diag.conflict_details.lower()


def test_diagnosis_branch_on_pace_and_insufficient_data():
    """Verify insufficient_data on early cycle below sample floor."""
    goal = _make_goal()
    obs = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at + timedelta(days=1),
        funnel_counts=FunnelStageCounts(
            roles_sourced=2,
            roles_assessed=2,
            applications=1,
            responses=0,
        ),
        conversion_rates=ConversionRates(),
        timeline=TimelineProgress(
            elapsed_days=1.0,
            remaining_days=89.0,
            total_days=90.0,
            pace_fraction=0.01,
        ),
        evidence_claims_count=10,
    )
    diag = diagnose(goal, obs)
    assert diag.category == DiagnosisCategory.ON_PACE
    assert diag.insufficient_data is True


def test_unsupported_llm_hypotheses_are_discarded():
    """LLM hypotheses without exact numeric backing in the observation are discarded."""
    from pydantic import BaseModel

    class LLMResp(BaseModel):
        hypotheses: list[RootCauseHypothesis]

    # Two hypotheses: one with real number (0.25 for assessed_to_applied), one with hallucinated 999.0
    valid_hyp = RootCauseHypothesis(
        cause=RootCauseKind.LOW_FIT_TARGETING,
        explanation="Low conversion rate from assessed to applied",
        supporting_metric_name="assessed_to_applied",
        supporting_metric_value=0.25,
    )
    hallucinated_hyp = RootCauseHypothesis(
        cause=RootCauseKind.INSUFFICIENT_VOLUME,
        explanation="Completely fabricated number",
        supporting_metric_name="ghost_metric",
        supporting_metric_value=999.0,
    )

    fake_llm = ScriptedLLM(LLMResp(hypotheses=[valid_hyp, hallucinated_hyp]))

    goal = _make_goal()
    obs = Observation(
        goal_id=goal.id,
        observed_at=goal.created_at + timedelta(days=22.5),
        funnel_counts=FunnelStageCounts(
            roles_sourced=32,
            roles_assessed=32,
            applications=8,
            responses=0,
        ),
        conversion_rates=ConversionRates(
            assessed_to_applied=0.25,
            applied_to_response=0.0,
        ),
        timeline=TimelineProgress(
            pace_fraction=0.25, elapsed_days=22.5, remaining_days=67.5, total_days=90.0
        ),
        evidence_claims_count=10,
    )

    diag = diagnose(goal, obs, llm=fake_llm)

    assert diag.category == DiagnosisCategory.STARVED
    assert len(diag.hypotheses) == 1
    assert diag.hypotheses[0].supporting_metric_value == 0.25
    assert diag.hypotheses[0].cause == RootCauseKind.LOW_FIT_TARGETING
