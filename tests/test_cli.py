"""Tests for the Pilot Command Line Interface (CLI)."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from pilot.cli.main import app
from pilot.db.models import EvidenceClaim, Goal, User
from pilot.extraction.extractor import ExtractionDroppedClaim, ExtractionResult
from pilot.goals.schemas import CompiledGoalDraft, FunnelAssumptions
from pilot.schemas.evidence import EvidenceClaimCreate, compute_claim_content_hash
from pilot.schemas.goal import TargetSpec

runner = CliRunner()


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Ensure OPENAI_API_KEY is configured for CLI tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-for-cli")


def test_cli_help_and_subcommand_helps():
    """Verify pilot --help and all subcommand --help commands exit 0 cleanly."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "Pilot: Goal-directed" in res.stdout

    for subcmd in [
        ["init", "--help"],
        ["ingest", "--help"],
        ["goal", "--help"],
        ["goal", "set", "--help"],
        ["show", "--help"],
    ]:
        sub_res = runner.invoke(app, subcmd)
        assert sub_res.exit_code == 0
        assert "Usage" in sub_res.stdout


def test_pilot_init_idempotent(db_session):
    """Test that pilot init successfully registers a user and repeated calls update idempotently."""
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    name = "CLI Candidate"

    # First invocation
    res1 = runner.invoke(
        app, ["init", "--email", email, "--name", name, "--github", "clicandidate"]
    )
    assert res1.exit_code == 0
    assert "User created successfully" in res1.stdout

    # Second invocation with updated name
    res2 = runner.invoke(
        app, ["init", "--email", email, "--name", "Updated Candidate", "--github", "clicandidate"]
    )
    assert res2.exit_code == 0
    assert "User updated successfully" in res2.stdout

    # Verify exactly one user in database
    users = db_session.scalars(select(User).where(User.email == email)).all()
    assert len(users) == 1
    assert users[0].full_name == "Updated Candidate"


def test_pilot_ingest_sample_resume_and_idempotency(db_session, mock_openai_env, monkeypatch):
    """Test pilot ingest extracts claims, displays audit summary, and re-running is idempotent."""
    # 1. Register candidate user
    email = f"ingest_{uuid.uuid4().hex[:8]}@example.com"
    user = User(email=email, full_name="Ingest Candidate")
    db_session.add(user)
    db_session.commit()

    # 2. Mock GroundedExtractor.extract to return deterministic claims and dropped record
    sample_resume = Path("tests/fixtures/sample_resume.txt")
    source_url = f"{sample_resume.resolve().as_uri()}#line=1;chars=0-50"
    claim_text = "Built real-time distributed feature store"
    content_hash = compute_claim_content_hash(claim_text, source_url)

    mock_claim = EvidenceClaimCreate(
        entity_type="user",
        entity_id=user.id,
        claim=claim_text,
        source="resume",
        source_url=source_url,
        source_excerpt="Built real-time distributed feature store",
        content_hash=content_hash,
        confidence=0.95,
    )
    mock_dropped = ExtractionDroppedClaim(
        claim="Invented Kubernetes single-handedly",
        source_excerpt="Worked with Kubernetes cluster",
        reason="Exaggerated unsupported claim",
    )
    mock_result = ExtractionResult(
        claims=[mock_claim],
        dropped=[mock_dropped],
        proposed_count=2,
    )

    monkeypatch.setattr(
        "pilot.cli.main.GroundedExtractor.extract",
        lambda self, spans, entity_type, entity_id, source: mock_result,
    )

    # First ingest run
    res1 = runner.invoke(app, ["ingest", "--resume", str(sample_resume)])
    assert res1.exit_code == 0
    assert "Extraction & Grounding Summary" in res1.stdout
    assert "Ungrounded Claims Dropped" in res1.stdout
    assert "Invented Kubernetes single-handedly" in res1.stdout
    assert "Successfully ingested and persisted 1 new evidence claims" in res1.stdout

    # Verify claim in database
    claims = db_session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)
    ).all()
    assert len(claims) == 1

    # Second ingest run (idempotent: 0 new rows written)
    res2 = runner.invoke(app, ["ingest", "--resume", str(sample_resume)])
    assert res2.exit_code == 0
    assert "Idempotent ingest: 0 new rows written" in res2.stdout


def test_pilot_ingest_image_only_pdf_exits_one(db_session, mock_openai_env):
    """Test that un-OCR'd image-only PDF exits 1 with a clean message and no traceback."""
    # Ensure active user exists
    user = User(email=f"pdf_{uuid.uuid4().hex[:8]}@example.com", full_name="PDF Candidate")
    db_session.add(user)
    db_session.commit()

    image_pdf = Path("tests/fixtures/image_only_resume.pdf")
    res = runner.invoke(app, ["ingest", "--resume", str(image_pdf)])
    assert res.exit_code == 1
    assert "Failed to read resume" in res.stdout
    assert "Traceback" not in res.stdout


def test_pilot_goal_set_past_deadline_exits_one(db_session, mock_openai_env):
    """Test that pilot goal set with a past deadline exits 1 without persisting a goal."""
    user = User(email=f"goal_{uuid.uuid4().hex[:8]}@example.com", full_name="Goal Candidate")
    db_session.add(user)
    db_session.commit()

    res = runner.invoke(
        app,
        ["goal", "set", "Land Applied AI Engineer role", "--deadline", "2020-01-01"],
    )
    assert res.exit_code == 1
    assert "Goal cannot be scheduled" in res.stdout
    assert "in the past" in res.stdout

    # Verify no goal was persisted
    goals = db_session.scalars(select(Goal).where(Goal.user_id == user.id)).all()
    assert len(goals) == 0


def test_pilot_goal_set_success_renders_spec_and_timeline(db_session, mock_openai_env, monkeypatch):
    """Test successful pilot goal set compiles and persists goal with rendered tables."""
    user = User(email=f"goal_ok_{uuid.uuid4().hex[:8]}@example.com", full_name="Goal Candidate OK")
    db_session.add(user)
    db_session.commit()

    # Mock StructuredLLMClient for GoalCompiler
    mock_draft = CompiledGoalDraft(
        target_spec=TargetSpec(
            must_have=["Applied AI Engineer", "Remote"],
            nice_to_have=["PyTorch", "vLLM"],
            unstated_but_real=["Can pass live coding screen", "Verifiable GitHub code"],
        ),
        success_criteria={"min_offers": 1, "min_salary": 140000},
        funnel_assumptions=FunnelAssumptions(
            application_to_response=0.10,
            response_to_interview=0.25,
            interview_to_offer=0.20,
            sourcing_to_application=0.50,
        ),
    )

    monkeypatch.setattr(
        "pilot.cli.main.OpenAIStructuredClient.complete_structured",
        lambda self, system_prompt, user_prompt, schema: mock_draft,
    )

    future_deadline = "2027-12-31"
    res = runner.invoke(
        app,
        [
            "goal",
            "set",
            "Land an Applied AI Engineer role by 2027",
            "--deadline",
            future_deadline,
            "--constraint",
            "max_applications_per_day=5",
        ],
    )
    assert res.exit_code == 0
    assert "Goal Compiled & Persisted" in res.stdout
    assert "Target Specification Breakdown" in res.stdout
    assert "Numeric Success Criteria" in res.stdout
    assert "Back-Solved Sub-Goal Timeline" in res.stdout

    goals = db_session.scalars(select(Goal).where(Goal.user_id == user.id)).all()
    assert len(goals) == 1
    assert goals[0].objective_text == "Land an Applied AI Engineer role by 2027"


def test_pilot_show_empty_db_prints_hint():
    """Test that pilot show with no user records exits 0 and prints registration hint."""
    res = runner.invoke(app, ["show"])
    assert res.exit_code == 0
    # Must print hint without traceback
    assert "Traceback" not in res.stdout
    assert (
        "Database is empty" in res.stdout
        or "No active goal found" in res.stdout
        or "No evidence claims found" in res.stdout
    )


def test_pilot_show_renders_active_goal_and_evidence_claims(db_session):
    """Test pilot show renders panels and tables when goal and claims exist."""
    user = User(email=f"show_{uuid.uuid4().hex[:8]}@example.com", full_name="Show Candidate")
    db_session.add(user)
    db_session.flush()

    goal = Goal(
        user_id=user.id,
        objective_text="Land Applied AI Engineer role",
        constraints_json={},
        target_spec={"must_have": ["AI Role"], "nice_to_have": [], "unstated_but_real": []},
        success_criteria={"min_offers": 1},
        sub_goals=[
            {
                "id": "sg_1",
                "title": "Portfolio",
                "metric_target": {"claims": 5},
                "deadline": "2027-01-01",
                "status": "pending",
            }
        ],
        deadline=datetime(2027, 6, 1, 0, 0, 0, tzinfo=UTC),
    )
    db_session.add(goal)

    claim = EvidenceClaim(
        entity_type="user",
        entity_id=user.id,
        claim="Trained LLaMA 3 fine-tuned model",
        source="resume",
        source_url="file:///resume.txt#line=1;chars=0-30",
        source_excerpt="Trained LLaMA 3 fine-tuned model",
        content_hash="mock_hash_123",
        confidence=0.92,
    )
    db_session.add(claim)
    db_session.commit()

    res = runner.invoke(app, ["show"])
    assert res.exit_code == 0
    assert "Active Goal Overview" in res.stdout
    assert "Land Applied AI Engineer role" in res.stdout
    assert "Verified Evidence Claims" in res.stdout
    assert "Trained LLaMA 3 fine-tuned model" in res.stdout
