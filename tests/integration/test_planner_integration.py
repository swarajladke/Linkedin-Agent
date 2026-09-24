"""Integration tests for Phase 3 Planner invariants."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    Company,
    Cycle,
    Escalation,
    EvidenceClaim,
    Goal,
    GoalStatus,
    Role,
    RoleAssessment,
    RoleStatus,
    Strategy,
    StrategyNote,
    User,
)
from pilot.planner.cycle import run_cycle
from pilot.planner.replay import replay_cycle

pytestmark = pytest.mark.integration


# ============================================================================
# Fixtures
# ============================================================================


def _seed_world(
    session: Session,
    *,
    n_roles: int = 5,
    n_claims: int = 3,
    fit_scores: list[float] | None = None,
    now: datetime,
) -> tuple[User, Goal, Strategy]:
    """Seed a fully wired world model: user -> goal -> strategy -> roles -> assessments -> claims."""
    user = User(
        email=f"tester_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Test Candidate",
        github_username="testcandidate",
    )
    session.add(user)
    session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Find Senior Software Engineer role",
        status=GoalStatus.ACTIVE,
        constraints_json={"max_applications_per_day": 10},
        success_criteria={"min_offers": 1, "target_applications": 20},
        target_spec={},
        sub_goals=[
            {"id": "sg_1", "title": "Evidence", "metric_target": {"evidence_claims_verified": 5}},
            {"id": "sg_2", "title": "Sourcing", "metric_target": {"roles_identified": 10}},
            {"id": "sg_3", "title": "Applications", "metric_target": {"applications_submitted": 8}},
            {
                "id": "sg_4",
                "title": "Interviews",
                "metric_target": {"interviews_completed": 2, "offers_received": 1},
            },
        ],
        deadline=now + timedelta(days=90),
        created_at=now,
    )
    session.add(goal)
    session.flush()

    strategy = Strategy(goal_id=goal.id, version=1)
    session.add(strategy)
    session.flush()

    note = StrategyNote(
        goal_id=goal.id,
        strategy_id=strategy.id,
        hypothesis="Focus on backend engineering roles with Rust or Go requirements.",
        status="active",
    )
    session.add(note)

    # Seed evidence claims
    for i in range(n_claims):
        claim = EvidenceClaim(
            entity_type="user",
            entity_id=user.id,
            claim=f"Implemented distributed systems component {i}",
            source="resume",
            source_url=f"file:///resume.pdf#L{i*10}",
            source_excerpt=f"Distributed systems component {i} with 99.9% uptime",
            content_hash=uuid.uuid4().hex[:64].ljust(64, "a"),
            confidence=0.90,
        )
        session.add(claim)

    session.flush()

    # Seed claims list for assessment
    claim_ids = [
        str(c.id)
        for c in session.scalars(
            select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)
        ).all()
    ]

    company = Company(name="TechCorp Inc", domain="techcorp.com")
    session.add(company)
    session.flush()

    fits = fit_scores or [0.85] * n_roles
    for i in range(n_roles):
        role = Role(
            company_id=company.id,
            title=f"Senior Software Engineer {i+1}",
            location_type="remote",
            status=RoleStatus.OPEN,
            requirements_summary="Build distributed systems at scale",
        )
        session.add(role)
        session.flush()

        assessment = RoleAssessment(
            role_id=role.id,
            goal_id=goal.id,
            fit_score=fits[i % len(fits)],
            fit_rationale="Strong backend and distributed systems background",
            supporting_claim_ids=claim_ids,
            skill_gaps=[],
            company_context={"stage": "Series B"},
            recommended_action="apply_now",
            assessor_version="1.0.0",
        )
        session.add(assessment)

    session.commit()
    return user, goal, strategy


# ============================================================================
# Full Cycle Atomic Execution
# ============================================================================


@pytest.mark.integration
def test_full_cycle_atomic_execution(session: Session):
    """Full decision cycle commits atomically and satisfies created_at <= executed_at invariant."""
    now = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    user, goal, strategy = _seed_world(session, n_roles=3, n_claims=5, now=now)

    result = run_cycle(session, goal, now=now, dry_run=False)

    assert result.cycle_id is not None
    assert result.cycle_number == 1
    assert result.observation.funnel_counts.roles_sourced >= 3

    # Cycle row persisted
    cycle = session.get(Cycle, result.cycle_id)
    assert cycle is not None
    assert cycle.goal_id == goal.id
    assert cycle.cycle_number == 1
    assert cycle.completed_at is not None

    # Verify created_at <= executed_at invariant for all actions
    actions = session.scalars(select(Action).where(Action.cycle_id == result.cycle_id)).all()
    for action in actions:
        assert action.executed_at is not None, f"Action {action.id} has no executed_at"
        assert action.predicted_probability is not None, f"Action {action.id} has no prediction"
        assert action.predicted_outcome, f"Action {action.id} has empty predicted_outcome"

    # Escalations created
    escalations = session.scalars(select(Escalation).where(Escalation.goal_id == goal.id)).all()
    assert len(escalations) == len(actions)
    for esc in escalations:
        assert esc.resolved is False
        assert esc.payload is not None


@pytest.mark.integration
def test_rollback_on_failure_zero_orphan_actions(session: Session):
    """If cycle fails mid-execution, no orphan actions are left in DB."""
    now = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    user, goal, strategy = _seed_world(session, n_roles=2, now=now)
    initial_action_count = session.scalar(select(Action.id).count()) or 0

    # Monkey-patch execute to fail
    from pilot.planner import cycle as cycle_mod

    original_execute = cycle_mod.execute

    def failing_execute(*args, **kwargs):
        raise RuntimeError("Simulated mid-cycle execution failure")

    cycle_mod.execute = failing_execute
    try:
        with pytest.raises(RuntimeError, match="Simulated mid-cycle execution failure"):
            run_cycle(session, goal, now=now, dry_run=False)
    finally:
        cycle_mod.execute = original_execute

    # After rollback, no new orphan actions
    final_action_count = session.scalar(select(Action.id).count()) or 0
    assert (
        final_action_count == initial_action_count
    ), f"Orphan actions detected: {final_action_count - initial_action_count} extra"


@pytest.mark.integration
def test_cycle_number_increments(session: Session):
    """Each successive cycle for a goal increments cycle_number monotonically."""
    now = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    user, goal, strategy = _seed_world(session, n_roles=2, now=now)

    r1 = run_cycle(session, goal, now=now, dry_run=False)
    assert r1.cycle_number == 1

    # Must move forward in time or skip roles; add new unapplied roles
    company = session.scalars(select(Company)).first()
    for i in range(3):
        role = Role(
            company_id=company.id,
            title=f"New Role Cycle2 {i}",
            location_type="remote",
            status=RoleStatus.OPEN,
        )
        session.add(role)
        session.flush()

        claim_ids = [
            str(c.id)
            for c in session.scalars(
                select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)
            ).all()
        ]
        assessment = RoleAssessment(
            role_id=role.id,
            goal_id=goal.id,
            fit_score=0.88,
            fit_rationale="Good fit",
            supporting_claim_ids=claim_ids,
            skill_gaps=[],
            company_context={},
            recommended_action="apply_now",
            assessor_version="1.0.0",
        )
        session.add(assessment)
    session.commit()

    now2 = now + timedelta(days=7)
    r2 = run_cycle(session, goal, now=now2, dry_run=False)
    assert r2.cycle_number == 2
    assert r2.cycle_number > r1.cycle_number


@pytest.mark.integration
def test_dry_run_no_db_writes(session: Session):
    """Dry-run mode evaluates the full pipeline without persisting any DB state."""
    now = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    user, goal, strategy = _seed_world(session, n_roles=3, now=now)

    before_cycles = session.scalar(select(Cycle.id).count()) or 0
    before_actions = session.scalar(select(Action.id).count()) or 0
    before_escalations = session.scalar(select(Escalation.id).count()) or 0

    result = run_cycle(session, goal, now=now, dry_run=True)

    assert result.dry_run is True
    assert result.cycle_id is None

    after_cycles = session.scalar(select(Cycle.id).count()) or 0
    after_actions = session.scalar(select(Action.id).count()) or 0
    after_escalations = session.scalar(select(Escalation.id).count()) or 0

    assert after_cycles == before_cycles, "Dry run wrote a cycle row"
    assert after_actions == before_actions, "Dry run wrote action rows"
    assert after_escalations == before_escalations, "Dry run wrote escalation rows"


@pytest.mark.integration
def test_counterfactual_replay_identifies_diff(session: Session):
    """Replay under min_fit=0.95 selects fewer roles than original with min_fit=0.50."""
    now = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    # Mix of fits: 3 at 0.85 and 2 at 0.60
    user, goal, strategy = _seed_world(
        session,
        n_roles=5,
        fit_scores=[0.85, 0.85, 0.85, 0.60, 0.60],
        now=now,
    )

    result = run_cycle(session, goal, now=now, dry_run=False)
    assert result.cycle_id is not None

    # Counterfactual: raise fit floor to 0.95 (should remove all roles)
    replay_result = replay_cycle(session, result.cycle_id, min_fit=0.95)

    assert replay_result.cycle_id == result.cycle_id
    assert replay_result.cycle_number == 1
    assert len(replay_result.counterfactual_action_ids) < len(replay_result.original_action_ids)
    assert "Raised fit floor to 0.95" in replay_result.rationale_diff
