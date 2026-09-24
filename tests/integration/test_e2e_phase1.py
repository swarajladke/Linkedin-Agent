"""End-to-end integration test validating the terminal state of the complete Phase 1 pipeline."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import EvidenceClaim, Goal, GoalStatus, Strategy, StrategyNote, User
from pilot.extraction.extractor import GroundedExtractor
from pilot.extraction.repository import upsert_evidence_claims
from pilot.goals.compiler import GoalCompiler
from pilot.goals.repository import persist_compiled_goal
from pilot.ingestion.reader import ResumeReader
from tests.integration.conftest import ScriptedLLM, assert_grounding_invariant

pytestmark = pytest.mark.integration

SAMPLE_RESUME = Path(__file__).parents[1] / "fixtures" / "sample_resume.txt"


def test_full_pipeline_e2e_terminal_state(session: Session, scripted_llm: ScriptedLLM):
    """Invariant: The full Phase 1 lifecycle from candidate ingestion through goal compilation and persistence produces a consistent terminal state with complete provenance and valid sub-goal horizons."""
    fixed_now = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
    goal_deadline = fixed_now + timedelta(days=60)

    # 1. Init user
    user = User(
        email="alex.e2e@example.com",
        full_name="Alex E2E Candidate",
        github_username="alexe2e",
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    # 2. Read resume fixture
    reader = ResumeReader()
    doc = reader.read(SAMPLE_RESUME)
    assert len(doc.spans) > 0

    # 3. Extract grounded claims via scripted LLM
    extractor = GroundedExtractor(llm=scripted_llm)
    extract_result = extractor.extract(
        spans=doc.spans,
        entity_type="user",
        entity_id=user.id,
        source="resume",
    )
    assert len(extract_result.claims) > 0
    assert extract_result.grounding_pass_rate == 1.0

    # 4. Upsert evidence claims
    upserted_count = upsert_evidence_claims(session, extract_result.claims)
    session.commit()
    assert upserted_count == len(extract_result.claims)

    # 5. Compile goal
    compiler = GoalCompiler(llm=scripted_llm, now=fixed_now)
    compiled_goal = compiler.compile(
        user_id=user.id,
        objective_text="Land Senior Applied AI Engineer role in Bangalore or Remote",
        constraints={"max_applications_per_day": 10},
        deadline=goal_deadline,
    )

    # 6. Persist compiled goal (creates Goal, Strategy v1, StrategyNote)
    persist_compiled_goal(session, compiled_goal)
    session.commit()

    # 7. Query back terminal state from Postgres
    persisted_claims = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)
    ).all()
    assert len(persisted_claims) == len(extract_result.claims)

    # Invariant: every evidence_claims row has non-empty source_excerpt, source_url, and 64-char hash
    for claim in persisted_claims:
        assert claim.source_excerpt is not None and len(claim.source_excerpt.strip()) > 0
        assert claim.source_url is not None and len(claim.source_url.strip()) > 0
        assert claim.content_hash is not None and len(claim.content_hash) == 64

    # Run the aggregate grounding invariant check over all persisted claims
    assert_grounding_invariant(persisted_claims, doc.spans)

    # Invariant: exactly one goals row with status = ACTIVE
    goals = session.scalars(select(Goal).where(Goal.user_id == user.id)).all()
    assert len(goals) == 1
    goal = goals[0]
    assert goal.status == GoalStatus.ACTIVE

    # Invariant: exactly one strategies row with version = 1 and parent_version_id IS NULL
    strategies = session.scalars(select(Strategy).where(Strategy.goal_id == goal.id)).all()
    assert len(strategies) == 1
    strategy = strategies[0]
    assert strategy.version == 1
    assert strategy.parent_version_id is None

    # Invariant: exactly one baseline strategy_notes row
    notes = session.scalars(
        select(StrategyNote).where(StrategyNote.strategy_id == strategy.id)
    ).all()
    assert len(notes) == 1
    note = notes[0]
    assert note.strategy_id == strategy.id
    assert len(note.hypothesis) > 0

    # Invariant: goal.sub_goals is non-empty and every deadline falls inside (created_at, deadline)
    assert goal.sub_goals is not None and len(goal.sub_goals) > 0
    ref_start = goal.created_at
    if ref_start.tzinfo is None and goal.deadline.tzinfo is not None:
        ref_start = ref_start.replace(tzinfo=UTC)

    for sg in goal.sub_goals:
        raw_deadline = sg["deadline"]
        sg_deadline = (
            datetime.fromisoformat(raw_deadline) if isinstance(raw_deadline, str) else raw_deadline
        )
        if sg_deadline.tzinfo is None and ref_start.tzinfo is not None:
            sg_deadline = sg_deadline.replace(tzinfo=UTC)

        msg = f"Sub-goal {sg['id']} deadline {sg_deadline} falls outside ({ref_start}, {goal.deadline})"
        assert ref_start < sg_deadline < goal.deadline, msg
