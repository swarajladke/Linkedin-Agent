"""Integration tests for Phase 4 Critic verification gate invariants."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.critic.voice import ingest_writing_sample
from pilot.db.models import (
    Action,
    Company,
    CriticReview,
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
    WritingSample,
)
from pilot.planner.execute import execute

pytestmark = pytest.mark.integration


# ============================================================================
# Invariant Checker
# ============================================================================
def assert_critic_escalation_invariant(session: Session) -> None:
    """
    Assert the core Phase 4 guarantee across all escalation records:
    Every Escalation payload artifact traces to a critic_reviews row with verdict = 'pass'.
    Nothing unchecked ever ships to a human.
    """
    escalations = session.scalars(select(Escalation)).all()
    for esc in escalations:
        if esc.action_id is None:
            raise AssertionError(f"Escalation {esc.id} is not linked to any Action.")

        passing_reviews = session.scalars(
            select(CriticReview).where(
                CriticReview.action_id == esc.action_id,
                CriticReview.verdict == "pass",
            )
        ).all()

        if not passing_reviews:
            raise AssertionError(
                f"Critical Invariant Violated: Escalation {esc.id} (Action {esc.action_id}) "
                "has no corresponding CriticReview with verdict = 'pass'."
            )


# ============================================================================
# Fixtures & Helpers
# ============================================================================
def _seed_critic_world(
    session: Session,
    *,
    claims_data: list[str] | None = None,
    banned_phrases: list[str] | None = None,
    now: datetime,
) -> tuple[User, Goal, Role, Action]:
    """Seed user, goal, strategy, claims, company, role, assessment, and action."""
    user = User(
        email=f"candidate_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Critic Test Candidate",
        github_username="critictester",
    )
    session.add(user)
    session.flush()

    if banned_phrases:
        # Ingest a writing sample that sets banned_phrases in the user's voice profile
        sample = WritingSample(
            user_id=user.id,
            content_type="text/plain",
            content_hash=uuid.uuid4().hex.ljust(64, "0"),
            raw_text="Clear technical writing without buzzwords.",
            spans=[],
            word_count=5,
            voice_profile={
                "banned_phrases": banned_phrases,
                "mean_sentence_length": 15.0,
                "contraction_rate": 0.0,
                "passive_voice_rate": 0.0,
                "total_words": 100,
            },
            created_at=now,
        )
        session.add(sample)
        session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Secure Senior Backend Role",
        status=GoalStatus.ACTIVE,
        constraints_json={"max_applications_per_day": 5},
        success_criteria={"min_offers": 1},
        target_spec={},
        sub_goals=[],
        deadline=now + timedelta(days=60),
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
        hypothesis="Target backend infrastructure engineering roles.",
        status="active",
    )
    session.add(note)

    default_claims = claims_data or ["Built distributed consensus protocol in Go"]
    claim_ids: list[str] = []
    for i, claim_text in enumerate(default_claims):
        claim = EvidenceClaim(
            entity_type="user",
            entity_id=user.id,
            claim=claim_text,
            source="resume",
            source_url=f"file:///resume.pdf#L{i * 10}",
            source_excerpt=claim_text,
            content_hash=uuid.uuid4().hex[:64].ljust(64, "b"),
            confidence=0.95,
        )
        session.add(claim)
        session.flush()
        claim_ids.append(str(claim.id))

    company = Company(name="Veritas Cloud", domain="veritas.example.com")
    session.add(company)
    session.flush()

    role = Role(
        company_id=company.id,
        title="Backend Infrastructure Engineer",
        location_type="remote",
        status=RoleStatus.OPEN,
        requirements_summary="Scale backend systems using Go and distributed databases.",
    )
    session.add(role)
    session.flush()

    assessment = RoleAssessment(
        role_id=role.id,
        goal_id=goal.id,
        fit_score=0.90,
        fit_rationale="Candidate exhibits strong backend alignment.",
        supporting_claim_ids=claim_ids,
        skill_gaps=[],
        company_context={"stage": "Series C"},
        recommended_action="apply_now",
        assessor_version="1.0.0",
    )
    session.add(assessment)
    session.flush()

    action = Action(
        goal_id=goal.id,
        strategy_id=strategy.id,
        action_type="generate_application_package",
        target_type="role",
        target_id=role.id,
        reason="Strong fit score 0.90",
        predicted_outcome="Advances to recruiter screen",
        predicted_probability=0.85,
        created_at=now,
    )
    session.add(action)
    session.flush()

    return user, goal, role, action


# ============================================================================
# Tests
# ============================================================================
def test_action_fails_twice_ends_dropped_zero_escalations(session: Session):
    """
    An action whose drafts fail twice ends 'drop', has exactly 2 critic_reviews rows,
    and zero Escalation rows.
    """
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)

    # Banned phrase 'engineer' is in summary and role title; cannot be auto-repaired
    user, goal, role, action = _seed_critic_world(
        session,
        claims_data=["Built high-performance storage engine"],
        banned_phrases=["engineer"],
        now=now,
    )

    result = execute(session, action, now=now)

    assert result.dropped is True
    assert result.critic_verdict == "drop"
    assert result.critic_attempts == 2
    assert result.escalation_id is None
    assert result.draft_package is None

    session.flush()

    # Exactly 2 reviews persisted (attempt 1 regenerate, attempt 2 drop)
    reviews = session.scalars(
        select(CriticReview)
        .where(CriticReview.action_id == action.id)
        .order_by(CriticReview.attempt.asc())
    ).all()
    assert len(reviews) == 2
    assert reviews[0].attempt == 1
    assert reviews[0].verdict == "regenerate"
    assert reviews[0].voice_passed is False
    assert reviews[1].attempt == 2
    assert reviews[1].verdict == "drop"
    assert reviews[1].voice_passed is False

    # Zero Escalations created
    escalations = session.scalars(select(Escalation).where(Escalation.action_id == action.id)).all()
    assert len(escalations) == 0

    # Action actual_outcome recorded structured drop
    assert action.actual_outcome is not None
    drop_data = json.loads(action.actual_outcome)
    assert drop_data.get("status") == "dropped"
    assert drop_data.get("attempts") == 2


def test_action_passes_on_attempt_two_with_one_escalation(session: Session):
    """
    An action that fails attempt 1 (banned phrase in one bullet) repairs on attempt 2,
    yielding 2 critic_reviews rows (second 'pass') and exactly 1 Escalation.
    """
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)

    # Claim 2 uses 'synergize' which is banned; Claim 1 is clean
    user, goal, role, action = _seed_critic_world(
        session,
        claims_data=[
            "Architected distributed Raft consensus library",
            "Helped synergize cross-functional agile teams",
        ],
        banned_phrases=["synergize"],
        now=now,
    )

    result = execute(session, action, now=now)

    assert result.dropped is False
    assert result.critic_verdict == "pass"
    assert result.critic_attempts == 2
    assert result.escalation_id is not None
    assert result.draft_package is not None

    session.flush()

    # Exactly 2 review attempts persisted
    reviews = session.scalars(
        select(CriticReview)
        .where(CriticReview.action_id == action.id)
        .order_by(CriticReview.attempt.asc())
    ).all()
    assert len(reviews) == 2
    assert reviews[0].attempt == 1
    assert reviews[0].verdict == "regenerate"
    assert reviews[0].voice_passed is False
    assert reviews[1].attempt == 2
    assert reviews[1].verdict == "pass"
    assert reviews[1].voice_passed is True

    # Exactly 1 Escalation created
    escalations = session.scalars(select(Escalation).where(Escalation.action_id == action.id)).all()
    assert len(escalations) == 1
    esc = escalations[0]
    assert esc.id == result.escalation_id
    assert esc.resolved is False


def test_critic_escalation_invariant(session: Session):
    """
    Database-level invariant test: every Escalation payload artifact traces to a
    critic_reviews row with verdict = 'pass'.
    """
    now = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)

    # 1. Clean run -> passes on attempt 1 -> 1 review (pass), 1 escalation
    _, _, _, action_clean = _seed_critic_world(
        session,
        claims_data=["Built key-value cache engine in C++"],
        banned_phrases=[],
        now=now,
    )
    res_clean = execute(session, action_clean, now=now)
    assert res_clean.dropped is False

    # 2. Defective run -> fails twice -> 2 reviews, 0 escalations
    _, _, _, action_bad = _seed_critic_world(
        session,
        claims_data=["Built backend API"],
        banned_phrases=["engineer"],
        now=now,
    )
    res_bad = execute(session, action_bad, now=now)
    assert res_bad.dropped is True

    session.flush()

    # Core invariant holds over all rows in database
    assert_critic_escalation_invariant(session)

    # Verify invariant detects unauthorized unreviewed escalation
    fake_esc = Escalation(
        id=uuid.uuid4(),
        goal_id=action_clean.goal_id,
        action_id=action_bad.id,  # bad action was dropped!
        reason="Unchecked ship attempt",
        payload={"unauthorized": True},
        resolved=False,
    )
    session.add(fake_esc)
    session.flush()

    with pytest.raises(AssertionError, match="Critical Invariant Violated"):
        assert_critic_escalation_invariant(session)

    # Clean up fake escalation
    session.delete(fake_esc)
    session.flush()


def test_writing_sample_idempotency(session: Session, tmp_path):
    """Re-ingesting the same writing sample twice produces exactly one row."""
    user = User(
        email=f"writer_{uuid.uuid4().hex[:6]}@example.com",
        full_name="Idempotency Test Candidate",
    )
    session.add(user)
    session.flush()

    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "I build software systems that scale cleanly. "
        "Simplicity and reliability are the core tenets of good engineering.",
        encoding="utf-8",
    )

    s1 = ingest_writing_sample(session, user.id, sample_file)
    session.commit()

    s2 = ingest_writing_sample(session, user.id, sample_file)
    session.commit()

    assert s1.id == s2.id
    assert s1.content_hash == s2.content_hash

    samples = session.scalars(select(WritingSample).where(WritingSample.user_id == user.id)).all()
    assert len(samples) == 1
    assert samples[0].word_count > 0
