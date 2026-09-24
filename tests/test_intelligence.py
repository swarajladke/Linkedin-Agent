"""Unit tests for role intelligence assessor, fit scoring, and grounding enforcement."""

import uuid
from typing import TypeVar

from pydantic import BaseModel

from pilot.db.models import Company, EvidenceClaim, Goal, Role
from pilot.extraction.llm import StructuredLLMClient
from pilot.intelligence.assessor import (
    LLMRoleSubSignals,
    RoleAssessor,
    compute_fit_score,
    derive_recommended_action,
)
from pilot.schemas.intelligence import SkillGap

T = TypeVar("T", bound=BaseModel)


class FakeStructuredLLM(StructuredLLMClient):
    """Fake LLM client returning a predefined structured evaluation."""

    def __init__(self, response: LLMRoleSubSignals) -> None:
        self.response = response

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        return self.response  # type: ignore[return-value]


def test_fit_score_pure_function():
    """Verify compute_fit_score is a pure deterministic function bounded in [0, 1]."""
    score1 = compute_fit_score(
        must_have_coverage=0.9,
        nice_to_have_coverage=0.8,
        location_compatibility=1.0,
        blocking_gaps_count=0,
        has_surviving_claims=True,
    )
    score2 = compute_fit_score(
        must_have_coverage=0.9,
        nice_to_have_coverage=0.8,
        location_compatibility=1.0,
        blocking_gaps_count=0,
        has_surviving_claims=True,
    )
    assert score1 == score2
    # 0.50*0.9 + 0.20*0.8 + 0.30*1.0 = 0.45 + 0.16 + 0.30 = 0.91
    assert score1 == 0.91

    # Bounded in [0, 1]
    assert (
        compute_fit_score(
            must_have_coverage=1.0,
            nice_to_have_coverage=1.0,
            location_compatibility=1.0,
            blocking_gaps_count=0,
            has_surviving_claims=True,
        )
        == 1.0
    )
    assert (
        compute_fit_score(
            must_have_coverage=0.0,
            nice_to_have_coverage=0.0,
            location_compatibility=0.0,
            blocking_gaps_count=5,
            has_surviving_claims=True,
        )
        == 0.0
    )


def test_fit_score_capped_when_no_surviving_claims():
    """Verify fit score is capped at 0.30 when not backed by at least one real claim."""
    score = compute_fit_score(
        must_have_coverage=1.0,
        nice_to_have_coverage=1.0,
        location_compatibility=1.0,
        blocking_gaps_count=0,
        has_surviving_claims=False,
    )
    assert score <= 0.30
    assert score == 0.30


def test_derive_recommended_action_rules():
    """Verify recommended_action derived by explicit business rules."""
    # When no blocking gaps
    assert derive_recommended_action(0.85, has_blocking_gaps=False) == "apply_now"
    assert derive_recommended_action(0.70, has_blocking_gaps=False) == "prepare_then_apply"
    assert derive_recommended_action(0.50, has_blocking_gaps=False) == "research_first"
    assert derive_recommended_action(0.20, has_blocking_gaps=False) == "skip"

    # Invariant: A role with a blocking gap NEVER returns apply_now
    assert derive_recommended_action(0.99, has_blocking_gaps=True) == "prepare_then_apply"
    assert derive_recommended_action(0.80, has_blocking_gaps=True) == "prepare_then_apply"
    assert derive_recommended_action(0.35, has_blocking_gaps=True) == "skip"


def test_assessor_discards_fabricated_claim_ids():
    """Verify fabricated supporting claim IDs are discarded and score recomputed."""
    real_claim_id = uuid.uuid4()
    fabricated_id = uuid.uuid4()

    fake_eval = LLMRoleSubSignals(
        fit_rationale="Strong match with Kubernetes and Python experience.",
        must_have_coverage=0.9,
        nice_to_have_coverage=0.8,
        location_compatibility=1.0,
        supporting_claim_ids=[str(real_claim_id), str(fabricated_id), "non-uuid-string"],
        skill_gaps=[],
        company_context={"name": "Canonical", "domain": "canonical.com"},
    )

    llm = FakeStructuredLLM(fake_eval)
    assessor = RoleAssessor(llm=llm, assessor_version="v1")

    company = Company(name="Canonical")
    role = Role(
        id=uuid.uuid4(),
        company_id=company.id,
        title="Infrastructure Engineer",
        location_type="remote",
        location="Remote",
        company=company,
    )
    goal = Goal(
        id=uuid.uuid4(),
        objective_text="Land Staff Infrastructure Engineer role",
        target_spec={"must_have": ["Python", "Kubernetes"]},
    )
    claims = [
        EvidenceClaim(
            id=real_claim_id,
            claim="Managed large-scale Kubernetes clusters",
            source="resume",
            source_url="resume.pdf",
            source_excerpt="Kubernetes clusters",
            content_hash="hash1",
        )
    ]

    assessment = assessor.assess(role=role, goal=goal, claims=claims)

    # Assert fabricated IDs were discarded
    assert assessment.supporting_claim_ids == [real_claim_id]
    assert assessment.fit_score == 0.91
    assert assessment.recommended_action == "apply_now"


def test_assessor_zero_surviving_claims_capped():
    """Verify that if LLM returns only hallucinated IDs, score is capped and action is not apply_now."""
    fake_eval = LLMRoleSubSignals(
        fit_rationale="Perfect match across all criteria.",
        must_have_coverage=1.0,
        nice_to_have_coverage=1.0,
        location_compatibility=1.0,
        supporting_claim_ids=[str(uuid.uuid4()), "fake-id-456"],
        skill_gaps=[],
        company_context={},
    )

    llm = FakeStructuredLLM(fake_eval)
    assessor = RoleAssessor(llm=llm, assessor_version="v1")

    role = Role(id=uuid.uuid4(), title="AI Architect", location_type="remote")
    goal = Goal(id=uuid.uuid4(), objective_text="AI Architect", target_spec={})
    claims = [
        EvidenceClaim(
            id=uuid.uuid4(),
            claim="Built ML models",
            source="github",
            source_url="url",
            source_excerpt="ML models",
            content_hash="h1",
        )
    ]

    assessment = assessor.assess(role=role, goal=goal, claims=claims)

    assert assessment.supporting_claim_ids == []
    # Score capped at 0.30
    assert assessment.fit_score <= 0.30
    assert assessment.recommended_action in ("research_first", "skip")


def test_blocking_gap_never_returns_apply_now():
    """Verify role with blocking must-have gap never returns apply_now regardless of high signals."""
    real_claim_id = uuid.uuid4()
    fake_eval = LLMRoleSubSignals(
        fit_rationale="Good match but missing mandatory C++ requirement.",
        must_have_coverage=0.9,
        nice_to_have_coverage=0.9,
        location_compatibility=1.0,
        supporting_claim_ids=[str(real_claim_id)],
        skill_gaps=[
            SkillGap(gap="Zero C++ systems programming evidence", severity="high", blocking=True)
        ],
        company_context={},
    )

    llm = FakeStructuredLLM(fake_eval)
    assessor = RoleAssessor(llm=llm, assessor_version="v1")

    role = Role(id=uuid.uuid4(), title="C++ Core Engineer", location_type="remote")
    goal = Goal(id=uuid.uuid4(), objective_text="Core Engineer", target_spec={})
    claims = [
        EvidenceClaim(
            id=real_claim_id,
            claim="Python systems",
            source="resume",
            source_url="resume.pdf",
            source_excerpt="Python systems",
            content_hash="h2",
        )
    ]

    assessment = assessor.assess(role=role, goal=goal, claims=claims)

    assert len(assessment.skill_gaps) == 1
    assert assessment.skill_gaps[0].blocking is True
    assert assessment.recommended_action != "apply_now"
    assert assessment.recommended_action == "prepare_then_apply"
