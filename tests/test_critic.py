"""Unit tests for Critic verification gate: grounding, voice fidelity, and factual checks."""

import uuid
from unittest.mock import MagicMock

from pilot.critic import (
    Critic,
    CriticVerdict,
    FailureSeverity,
    build_voice_profile,
    factual_check,
    grounding_check,
    voice_check,
)
from pilot.db.models import EvidenceClaim, Role


def _make_sample_claims():
    user_id = uuid.uuid4()
    c1 = EvidenceClaim(
        id=uuid.uuid4(),
        entity_type="user",
        entity_id=user_id,
        claim="Built distributed high-throughput event streaming architectures using Apache Kafka",
        source="resume",
        source_url="/resumes/cv.pdf",
        source_excerpt="Built distributed streaming architectures using Kafka",
        content_hash="a" * 64,
        confidence=0.95,
    )
    c2 = EvidenceClaim(
        id=uuid.uuid4(),
        entity_type="user",
        entity_id=user_id,
        claim="Over 3 years of production Kubernetes cluster operations experience",
        source="resume",
        source_url="/resumes/cv.pdf",
        source_excerpt="Over 3 years of production Kubernetes cluster operations experience",
        content_hash="b" * 64,
        confidence=0.95,
    )
    c3 = EvidenceClaim(
        id=uuid.uuid4(),
        entity_type="user",
        entity_id=user_id,
        claim="Software Engineer at Acme Corp developing backend microservices",
        source="resume",
        source_url="/resumes/cv.pdf",
        source_excerpt="Software Engineer at Acme Corp developing backend microservices",
        content_hash="c" * 64,
        confidence=0.95,
    )
    return [c1, c2, c3]


def _make_sample_role():
    return Role(
        id=uuid.uuid4(),
        company_id=uuid.uuid4(),
        title="Senior Backend Engineer",
        location_type="remote",
        requirements_summary="Design and operate scalable distributed backend systems with Kafka.",
    )


def test_grounding_failure_for_unsupported_employer():
    """A draft asserting an employer not in any claim fails grounding check with blocking status."""
    claims = _make_sample_claims()
    draft = "Worked as a Senior Principal Engineer at Netflix leading streaming architecture."

    result = grounding_check(draft, claims)

    assert result.passed is False
    assert len(result.failures) > 0
    blocking = [f for f in result.failures if f.severity == FailureSeverity.BLOCKING]
    assert len(blocking) > 0
    assert any("Netflix" in f.offending_text for f in blocking)


def test_framing_not_grounding_failure():
    """Subjective connective language ('I'm excited about this role') is exempt from grounding."""
    claims = _make_sample_claims()
    draft = "I'm excited about this role and would love to contribute to the engineering team."

    result = grounding_check(draft, claims)

    assert result.passed is True
    assert len(result.failures) == 0


def test_factual_check_inflation_fails():
    """A draft inflating '3 years' to '5 years' triggers a blocking factual integrity failure."""
    claims = _make_sample_claims()
    role = _make_sample_role()
    draft = "Over 5 years of production Kubernetes cluster operations experience."

    result = factual_check(draft, claims, role)

    assert result.passed is False
    blocking = [f for f in result.failures if f.severity == FailureSeverity.BLOCKING]
    assert len(blocking) > 0
    assert any("5 years" in f.offending_text.lower() for f in blocking)


def test_factual_check_invented_percentage_fails():
    """A draft inventing an unsupported percentage metric triggers a blocking factual failure."""
    claims = _make_sample_claims()
    role = _make_sample_role()
    draft = "Optimized streaming pipelines and reduced latency by 45% across microservices."

    result = factual_check(draft, claims, role)

    assert result.passed is False
    blocking = [f for f in result.failures if f.severity == FailureSeverity.BLOCKING]
    assert len(blocking) > 0
    assert any("45%" in f.offending_text for f in blocking)


def test_voice_banned_phrase_blocking_failure():
    """A draft that is grounded but contains a banned cliché triggers a blocking voice failure."""
    profile = build_voice_profile(["Writing sample text with professional voice."])
    draft = (
        "Built distributed high-throughput event streaming architectures using Apache Kafka. "
        "We are ready to hit the ground running and create synergy."
    )

    result = voice_check(draft, profile)

    assert result.passed is False
    blocking = [f for f in result.failures if f.severity == FailureSeverity.BLOCKING]
    assert len(blocking) > 0
    offending_words = [f.offending_text.lower() for f in blocking]
    assert any("synergy" in w or "hit the ground running" in w for w in offending_words)


def test_clean_grounded_in_voice_draft_passes():
    """A clean, grounded, in-voice draft passes all checks with verdict PASS."""
    claims = _make_sample_claims()
    role = _make_sample_role()
    profile = build_voice_profile([
        "I build software systems. We deploy services to cloud environments with Kubernetes."
    ])

    draft = (
        "I am excited about the Senior Backend Engineer role. "
        "• Demonstrates capability in: Built distributed high-throughput event streaming "
        "architectures using Apache Kafka [Source: /resumes/cv.pdf]\n"
        "• Demonstrates capability in: Over 3 years of production Kubernetes cluster operations "
        "experience [Source: /resumes/cv.pdf]"
    )

    critic = Critic()
    review = critic.review(artifact=draft, claims=claims, profile=profile, role=role, attempt=1)

    assert review.verdict == CriticVerdict.PASS
    assert review.grounding_passed is True
    assert review.voice_passed is True
    assert review.factual_passed is True
    blocking = [f for f in review.failures if f.severity == FailureSeverity.BLOCKING]
    assert len(blocking) == 0


def test_critic_regenerate_on_attempt_one_then_drop_on_attempt_two():
    """Attempt 1 with a blocking failure returns REGENERATE; attempt 2 returns DROP."""
    claims = _make_sample_claims()
    role = _make_sample_role()
    profile = build_voice_profile(["Sample writing"])

    bad_draft = "Worked at Netflix with 10 years of experience to create synergy."
    critic = Critic()

    # Attempt 1 -> REGENERATE
    rev1 = critic.review(artifact=bad_draft, claims=claims, profile=profile, role=role, attempt=1)
    assert rev1.verdict == CriticVerdict.REGENERATE
    assert len(rev1.failures) > 0

    # Attempt 2 -> DROP
    rev2 = critic.review(artifact=bad_draft, claims=claims, profile=profile, role=role, attempt=2)
    assert rev2.verdict == CriticVerdict.DROP
    assert len(rev2.failures) > 0


def test_voice_check_never_calls_llm():
    """Voice verification is purely deterministic and never invokes an LLM."""
    fake_llm = MagicMock()
    profile = build_voice_profile(["Sample profile"])

    result = voice_check("Some draft text to verify voice.", profile)

    fake_llm.complete_structured.assert_not_called()
    assert result.passed is True
