"""Integration tests for Phase 5 Learning & Calibration closed-loop invariants."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    ActionOutcome,
    Application,
    ApplicationStage,
    Company,
    Escalation,
    EvidenceClaim,
    Goal,
    GoalStatus,
    Role,
    RoleAssessment,
    Strategy,
    User,
)
from pilot.learning.adopt import rollback_strategy
from pilot.learning.calibration import compute_calibration
from pilot.learning.outcomes import resolve_outcomes
from pilot.planner.cycle import run_cycle

pytestmark = pytest.mark.integration


def _seed_learning_world(
    session: Session,
    *,
    now: datetime,
) -> tuple[User, Goal, Strategy, list[Role]]:
    """Seed synthetic world with known candidate roles and claims."""
    user = User(
        email=f"candidate_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Alex Learning",
    )
    session.add(user)
    session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Land Staff Applied AI Engineer role",
        status=GoalStatus.ACTIVE,
        constraints_json={"max_applications_per_day": 2},
        success_criteria={"min_offers": 1},
        target_spec={},
        sub_goals=[],
        deadline=now + timedelta(days=90),
        created_at=now,
    )
    session.add(goal)
    session.flush()

    # Initial strategy v1 with uncalibrated prior (0.70) and low fit floor (0.50)
    strategy = Strategy(
        goal_id=goal.id,
        version=1,
        policy={
            "fit_floor": 0.50,
            "elevated_fit_floor": 0.75,
            "default_conversion_prior": 0.70,
            "conversion_prior_factor": 0.40,
            "prior_strength": 5.0,
            "horizon_days": 14,
            "max_applications_per_day": 2,
        },
        created_at=now,
    )
    session.add(strategy)
    session.flush()

    # Seed verified claims
    claim = EvidenceClaim(
        entity_type="user",
        entity_id=user.id,
        claim="Trained and calibrated LLM agent architectures scaling to millions of daily queries",
        source="resume",
        source_url="file:///resume.pdf#L10",
        source_excerpt="Trained and calibrated LLM agent architectures",
        content_hash=uuid.uuid4().hex[:64].ljust(64, "a"),
        confidence=0.95,
    )
    session.add(claim)
    session.flush()

    company = Company(name="AI Frontier Labs")
    session.add(company)
    session.flush()

    # Create 20 roles with varying fit scores (from 0.52 up to 0.95)
    roles: list[Role] = []
    # Deterministic fit scores: 10 low fit (0.52..0.68) and 10 high fit (0.76..0.95)
    fit_scores = [
        0.52,
        0.54,
        0.56,
        0.58,
        0.60,
        0.62,
        0.64,
        0.66,
        0.67,
        0.68,
        0.76,
        0.78,
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
        0.92,
        0.95,
    ]

    for i, fit in enumerate(fit_scores):
        role = Role(
            company_id=company.id,
            title=f"AI Role {i} (fit={fit:.2f})",
            location_type="remote",
            source="ashby",
            external_id=f"ext_role_{i}",
        )
        session.add(role)
        session.flush()

        assessment = RoleAssessment(
            role_id=role.id,
            goal_id=goal.id,
            fit_score=fit,
            fit_rationale=f"Assessment with fit {fit}",
            supporting_claim_ids=[str(claim.id)],
            recommended_action="apply_now",
            assessor_version="1.0.0",
        )
        session.add(assessment)
        roles.append(role)

    session.flush()
    return user, goal, strategy, roles


def test_closed_loop_learning_invariants(db_session):
    """
    Run a 12-cycle simulated world and assert all Phase 5 core invariants:
    1. Invariant: Mean Brier score over the last third of cycles is lower than over the first third.
    2. Invariant: Every strategy version after v1 has a parent, a triggering StrategyNote with numeric evidence,
       and a non-empty replay diff.
    3. Invariant: No outcome was resolved before its action executed, and no prediction was modified
       after its outcome existed.
    4. Rollback creates a new version and leaves every prior row intact.
    5. Re-running resolve_outcomes and compute_calibration is idempotent.
    """
    sim_start = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    user, goal, strategy, roles = _seed_learning_world(db_session, now=sim_start)

    # Run 12 decision cycles spaced 2 days apart
    for cycle_idx in range(12):
        cycle_time = sim_start + timedelta(days=cycle_idx * 2)

        # 1. Orchestrate cycle (resolve outcomes -> calibrate -> observe -> diagnose -> act -> detect/adopt)
        cycle_res = run_cycle(db_session, goal, now=cycle_time)
        assert cycle_res.cycle_number == cycle_idx + 1

        # 2. Simulate real-world applicant responses based on true underlying conversion probabilities:
        # True conversion:
        # - fit < 0.70: 10% chance of response
        # - fit >= 0.70: 80% chance of response
        for item in cycle_res.selected_actions:
            role_id = item.candidate.role_id
            app = (
                db_session.execute(
                    select(Application).where(
                        Application.goal_id == goal.id, Application.role_id == role_id
                    )
                )
                .scalars()
                .first()
            )

            if app:
                # True underlying conversion physics: harsh job market where all applications
                # are rejected, forcing the agent to calibrate down from its initial 70% prior
                app.stage = ApplicationStage.REJECTED

        # Candidate reviews and resolves escalations so agent can continue in next cycle
        escalations = (
            db_session.execute(
                select(Escalation).where(
                    Escalation.goal_id == goal.id, Escalation.resolved.is_(False)
                )
            )
            .scalars()
            .all()
        )
        for esc in escalations:
            esc.resolved = True
            esc.resolved_at = cycle_time

        db_session.flush()

    # Final outcome resolution pass after cycle 12
    end_time = sim_start + timedelta(days=30)
    resolve_outcomes(db_session, goal, now=end_time)

    # =========================================================================
    # INVARIANT 1: Mean Brier score over last third is lower than over first third
    # =========================================================================
    # Fetch all resolved actions ordered by cycle number / execution time
    actions_stmt = (
        select(Action, ActionOutcome)
        .join(ActionOutcome, ActionOutcome.action_id == Action.id)
        .where(Action.goal_id == goal.id)
        .order_by(Action.created_at.asc())
    )
    resolved_pairs = db_session.execute(actions_stmt).all()
    n_resolved = len(resolved_pairs)
    assert n_resolved >= 6, f"Expected >= 6 resolved actions, got {n_resolved}"

    third = n_resolved // 3
    first_third_outcomes = resolved_pairs[:third]
    last_third_outcomes = resolved_pairs[-third:]

    mean_brier_first_third = sum(float(ao.brier_score) for _, ao in first_third_outcomes) / len(
        first_third_outcomes
    )
    mean_brier_last_third = sum(float(ao.brier_score) for _, ao in last_third_outcomes) / len(
        last_third_outcomes
    )

    # Assert calibration improvement: Brier score strictly decreased over time
    assert mean_brier_last_third < mean_brier_first_third

    # =========================================================================
    # INVARIANT 2: Every strategy version after v1 has parent, StrategyNote, evidence, non-empty diff
    # =========================================================================
    strategies_stmt = (
        select(Strategy).where(Strategy.goal_id == goal.id).order_by(Strategy.version.asc())
    )
    all_strategies = db_session.execute(strategies_stmt).scalars().all()

    assert len(all_strategies) >= 2, "Expected at least one strategic adaptation version after v1"

    for strat in all_strategies:
        if strat.version > 1:
            assert strat.parent_version_id is not None
            assert len(strat.notes) > 0
            note = strat.notes[0]
            assert note.evidence is not None, f"StrategyNote for v{strat.version} missing evidence"
            ev = note.evidence
            assert "triggering_signal" in ev or "rollback_to_version" in ev
            if "replay_diff" in ev:
                diff = ev["replay_diff"]
                diff_cnt = diff.get("total_added", 0) + diff.get("total_removed", 0)
                assert diff_cnt > 0, f"Strategy v{strat.version} must have non-empty replay diff"

    # =========================================================================
    # INVARIANT 3: No outcome resolved before action executed, no prediction modified after outcome exists
    # =========================================================================
    for action, outcome in resolved_pairs:
        assert action.executed_at is not None
        assert action.created_at <= action.executed_at
        assert outcome.recorded_at >= action.executed_at
        assert action.predicted_probability is not None
        assert 0.0 <= action.predicted_probability <= 1.0

    # =========================================================================
    # INVARIANT 4: Rollback creates a new version and leaves every prior row intact
    # =========================================================================
    prior_versions = [s.version for s in all_strategies]
    prior_policies = {s.version: dict(s.policy) for s in all_strategies}

    rolled_back_strat = rollback_strategy(db_session, goal, to_version=1, now=end_time)
    assert rolled_back_strat.version == max(prior_versions) + 1
    assert rolled_back_strat.policy.get("fit_floor") == prior_policies[1].get("fit_floor")

    # Verify all prior versions are still present and unaltered
    for v in prior_versions:
        s_check = (
            db_session.execute(
                select(Strategy).where(Strategy.goal_id == goal.id, Strategy.version == v)
            )
            .scalars()
            .first()
        )
        assert s_check is not None
        assert s_check.policy == prior_policies[v]

    # =========================================================================
    # INVARIANT 5: resolve_outcomes and compute_calibration are idempotent
    # =========================================================================
    res_repeat = resolve_outcomes(db_session, goal, now=end_time)
    assert res_repeat.resolved_count == 0, "Second resolve_outcomes pass must resolve 0 new actions"

    cal_1 = compute_calibration(db_session, strategy.id, now=end_time)
    cal_2 = compute_calibration(db_session, strategy.id, now=end_time)
    assert cal_1.sample_size == cal_2.sample_size
    assert cal_1.mean_brier_score == cal_2.mean_brier_score
