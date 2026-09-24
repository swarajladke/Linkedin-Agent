"""Integration tests asserting idempotency of ingestion, timestamp refreshes, and multi-source provenance."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from pilot.cli.main import app
from pilot.config import get_settings
from pilot.db.models import EvidenceClaim, User
from pilot.extraction.extractor import GroundedExtractor
from pilot.extraction.repository import upsert_evidence_claims
from pilot.ingestion.github import GitHubProfile, GitHubRepo, GitHubUserData
from pilot.ingestion.reader import ResumeReader
from pilot.schemas.evidence import EvidenceClaimCreate, compute_claim_content_hash
from tests.integration.conftest import ScriptedLLM, phase1_pipeline

pytestmark = pytest.mark.integration

SAMPLE_RESUME = Path(__file__).parents[1] / "fixtures" / "sample_resume.txt"


@pytest.fixture
def github_fixture() -> GitHubUserData:
    """Provide realistic structured GitHub user data."""
    now = datetime.now(UTC)
    profile = GitHubProfile(
        username="alexcandidate",
        name="Alex Candidate",
        bio="AI Engineer passionate about LLM inference and agentic workflows.",
        company="AI Research Lab",
        location="Bangalore",
        public_repos=1,
        created_at=now,
        html_url="https://github.com/alexcandidate",
    )
    repo = GitHubRepo(
        name="pilot-agent",
        description="Goal-directed autonomous career agent in Python.",
        html_url="https://github.com/alexcandidate/pilot-agent",
        language="Python",
        languages={"Python": 80000, "SQL": 15000},
        topics=["agent", "ai"],
        created_at=now,
        pushed_at=now,
        fork=False,
        archived=False,
        readme="# Pilot Agent\n\nBuilt with Python 3.11 and PostgreSQL with pgvector.",
        readme_url="https://github.com/alexcandidate/pilot-agent#readme",
        user_commit_count=42,
    )
    return GitHubUserData(
        username="alexcandidate",
        profile=profile,
        repos=[repo],
        fetched_at=now,
        source_url=profile.html_url,
    )


def test_ingest_path_three_runs_identical_counts_and_hashes(
    session: Session,
    seeded_user: User,
    github_fixture: GitHubUserData,
):
    """Invariant: Ingesting the identical source document set arbitrarily many times must produce zero duplicate rows and preserve the exact set of content hashes across all executions."""
    llm = ScriptedLLM()

    # Run 1
    phase1_pipeline(
        session,
        user=seeded_user,
        resume_path=SAMPLE_RESUME,
        github_data=github_fixture,
        llm=llm,
        compile_goal=False,
    )
    claims_1 = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    count_1 = len(claims_1)
    assert count_1 > 0
    hashes_1 = {c.content_hash for c in claims_1}

    # Run 2
    phase1_pipeline(
        session,
        user=seeded_user,
        resume_path=SAMPLE_RESUME,
        github_data=github_fixture,
        llm=llm,
        compile_goal=False,
    )
    claims_2 = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    assert len(claims_2) == count_1
    assert {c.content_hash for c in claims_2} == hashes_1

    # Run 3
    phase1_pipeline(
        session,
        user=seeded_user,
        resume_path=SAMPLE_RESUME,
        github_data=github_fixture,
        llm=llm,
        compile_goal=False,
    )
    claims_3 = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    assert len(claims_3) == count_1
    assert {c.content_hash for c in claims_3} == hashes_1


def test_reingest_advances_verified_at_while_created_at_is_immutable(
    session: Session,
    seeded_user: User,
):
    """Invariant: Re-ingestion of existing claims represents a verification refresh rather than an insertion, leaving created_at immutable while advancing verified_at."""
    reader = ResumeReader()
    doc = reader.read(SAMPLE_RESUME)
    llm = ScriptedLLM()
    extractor = GroundedExtractor(llm=llm)

    extracted = extractor.extract(
        spans=doc.spans,
        entity_type="user",
        entity_id=seeded_user.id,
        source="resume",
    )
    assert len(extracted.claims) > 0

    # Initial ingestion
    upsert_evidence_claims(session, extracted.claims)
    session.commit()

    initial_claims = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    initial_created = {c.id: c.created_at for c in initial_claims}

    # Artificially roll back verified_at by 1 hour to simulate the passage of time without sleeping
    session.execute(
        text("UPDATE evidence_claims SET verified_at = verified_at - INTERVAL '1 hour'")
    )
    session.commit()

    rolled_claims = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    rolled_verified = {c.id: c.verified_at for c in rolled_claims}

    # Re-ingest the exact same claims
    upsert_evidence_claims(session, extracted.claims)
    session.commit()

    refreshed_claims = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()

    for claim in refreshed_claims:
        # created_at must be strictly immutable
        assert claim.created_at == initial_created[claim.id]
        # verified_at must have advanced beyond the rolled-back timestamp
        assert claim.verified_at > rolled_verified[claim.id]


def test_identical_claim_text_from_different_sources_creates_distinct_rows(
    session: Session,
    seeded_user: User,
):
    """Invariant: The same factual claim originating from distinct sources (e.g. resume vs GitHub) produces distinct content hashes and rows, preserving provenance without erroneous deduplication."""
    claim_text = "Designed and deployed low-latency inference pipelines using vLLM"
    source_url_resume = "file:///resume.txt#line=10;excerpt_chars=0-50"
    source_url_github = "https://github.com/alexcandidate/repo#profile;field=bio"

    hash_resume = compute_claim_content_hash(claim=claim_text, source_url=source_url_resume)
    hash_github = compute_claim_content_hash(claim=claim_text, source_url=source_url_github)

    # Invariant: different source_url must yield different hash
    assert hash_resume != hash_github

    claim_1 = EvidenceClaimCreate(
        entity_type="user",
        entity_id=seeded_user.id,
        claim=claim_text,
        source="resume",
        source_url=source_url_resume,
        source_excerpt="low-latency inference pipelines using vLLM",
        content_hash=hash_resume,
        confidence=0.95,
    )
    claim_2 = EvidenceClaimCreate(
        entity_type="user",
        entity_id=seeded_user.id,
        claim=claim_text,
        source="github",
        source_url=source_url_github,
        source_excerpt="low-latency inference pipelines using vLLM",
        content_hash=hash_github,
        confidence=0.90,
    )

    upserted = upsert_evidence_claims(session, [claim_1, claim_2])
    session.commit()
    assert upserted == 2

    persisted = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    assert len(persisted) == 2

    sources = {c.source for c in persisted}
    urls = {c.source_url for c in persisted}
    hashes = {c.content_hash for c in persisted}

    assert sources == {"resume", "github"}
    assert urls == {source_url_resume, source_url_github}
    assert hashes == {hash_resume, hash_github}


def test_pilot_init_three_runs_single_user(session: Session, monkeypatch):
    """Invariant: Repeated invocations of user registration via CLI must be strictly idempotent, resulting in exactly one user record in the database."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key")
    get_settings.cache_clear()

    runner = CliRunner()
    email = "triple.init@example.com"
    name = "Triple Candidate"

    for _ in range(3):
        res = runner.invoke(app, ["init", "--email", email, "--name", name, "--github", "alex"])
        assert res.exit_code == 0

    users = session.scalars(select(User).where(User.email == email)).all()
    assert len(users) == 1
    assert users[0].email == email
    assert users[0].full_name == name
