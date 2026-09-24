"""Integration tests for role assessment versioning, in-place updates, and claim grounding invariants."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Company, EvidenceClaim, Goal, Role, RoleAssessment, User
from pilot.intelligence.assessor import RoleAssessor
from pilot.intelligence.repository import upsert_assessments
from pilot.schemas.intelligence import SkillGap
from tests.test_intelligence import FakeStructuredLLM, LLMRoleSubSignals

pytestmark = pytest.mark.integration


def assert_assessment_grounding_invariant(
    session: Session,
    assessment: RoleAssessment,
    expected_user_id: uuid.UUID,
) -> None:
    """
    Aggregate Invariant:
    Every persisted supporting_claim_ids entry must resolve to a real evidence_claims
    row belonging to the exact same candidate user.
    """
    assert len(assessment.supporting_claim_ids) > 0, "No claims cited to verify grounding"
    claim_uuids = [uuid.UUID(cid) for cid in assessment.supporting_claim_ids]

    stmt = select(EvidenceClaim).where(EvidenceClaim.id.in_(claim_uuids))
    db_claims = session.scalars(stmt).all()

    assert (
        len(db_claims) == len(claim_uuids)
    ), f"Grounding violation: {len(claim_uuids) - len(db_claims)} cited claim IDs do not exist in DB!"

    for claim in db_claims:
        assert (
            claim.entity_id == expected_user_id
        ), f"Grounding violation: claim {claim.id} belongs to entity {claim.entity_id}, expected {expected_user_id}!"


def test_assessment_versioning_and_grounding_invariants(session: Session, seeded_user: User):
    """
    Integration Invariant:
    - Re-assessing under the same assessor_version updates in place.
    - Bumping assessor_version adds a new row and preserves the previous one intact for replay.
    - Every persisted supporting_claim_ids entry resolves to a real evidence_claims row belonging to the user.
    """
    # 1. Seed candidate evidence claims
    claim1 = EvidenceClaim(
        id=uuid.uuid4(),
        entity_type="user",
        entity_id=seeded_user.id,
        claim="Architected distributed event streaming using Kafka and Python",
        source="resume",
        source_url="file:///resume.pdf#page=1",
        source_excerpt="Kafka and Python",
        content_hash="claim_hash_kafka_1",
        confidence=0.95,
    )
    claim2 = EvidenceClaim(
        id=uuid.uuid4(),
        entity_type="user",
        entity_id=seeded_user.id,
        claim="Built Kubernetes operators in Go and Python",
        source="github",
        source_url="https://github.com/alex/k8s-operator",
        source_excerpt="Kubernetes operators in Go",
        content_hash="claim_hash_k8s_2",
        confidence=0.90,
    )
    session.add_all([claim1, claim2])
    session.commit()

    # 2. Seed active goal
    goal = Goal(
        id=uuid.uuid4(),
        user_id=seeded_user.id,
        objective_text="Land Staff Infrastructure Engineer role",
        constraints_json={"remote_only": True},
        target_spec={"must_have": ["Python", "Kubernetes"]},
        success_criteria={"min_base_salary": 220000},
        sub_goals=[],
        deadline=datetime.now(UTC),
    )
    session.add(goal)

    # 3. Seed company and role
    company = Company(id=uuid.uuid4(), name="Platform Co", domain="platform.io")
    session.add(company)

    role = Role(
        id=uuid.uuid4(),
        company_id=company.id,
        title="Staff Infrastructure Engineer",
        location_type="remote",
        location="Remote, US",
        source="greenhouse",
        external_id="gh-plat-101",
        requirements_summary="Lead infrastructure automation with Python, Go, and Kubernetes.",
    )
    session.add(role)
    session.commit()

    # 4. Assess under assessor_version='v1'
    fake_eval_v1 = LLMRoleSubSignals(
        fit_rationale="Strong match with Kubernetes and Python experience.",
        must_have_coverage=0.9,
        nice_to_have_coverage=0.8,
        location_compatibility=1.0,
        supporting_claim_ids=[str(claim1.id), str(claim2.id)],
        skill_gaps=[],
        company_context={"name": "Platform Co"},
    )
    assessor_v1 = RoleAssessor(llm=FakeStructuredLLM(fake_eval_v1), assessor_version="v1")
    assess_v1 = assessor_v1.assess(role=role, goal=goal, claims=[claim1, claim2])

    persisted_v1 = upsert_assessments(session, [assess_v1])
    session.commit()

    assert len(persisted_v1) == 1
    assert persisted_v1[0].assessor_version == "v1"
    assert persisted_v1[0].fit_score == 0.91

    # Verify aggregate grounding invariant on v1
    assert_assessment_grounding_invariant(session, persisted_v1[0], seeded_user.id)

    # 5. Re-assess under the SAME assessor_version='v1' (update in place)
    fake_eval_v1_updated = LLMRoleSubSignals(
        fit_rationale="Updated rationale: Exceptional fit with verified cloud deployments.",
        must_have_coverage=0.95,
        nice_to_have_coverage=0.85,
        location_compatibility=1.0,
        supporting_claim_ids=[str(claim1.id), str(claim2.id)],
        skill_gaps=[],
        company_context={"name": "Platform Co"},
    )
    assess_v1_updated = RoleAssessor(
        llm=FakeStructuredLLM(fake_eval_v1_updated), assessor_version="v1"
    ).assess(role=role, goal=goal, claims=[claim1, claim2])

    persisted_v1_updated = upsert_assessments(session, [assess_v1_updated])
    session.commit()
    assert len(persisted_v1_updated) == 1

    # Invariant: Must update in place (total rows remains 1)
    all_assessments = session.scalars(select(RoleAssessment)).all()
    assert len(all_assessments) == 1
    assert all_assessments[0].id == persisted_v1[0].id
    assert "Updated rationale" in all_assessments[0].fit_rationale
    assert (
        all_assessments[0].fit_score == 0.945
    )  # 0.5*0.95 + 0.2*0.85 + 0.3*1.0 = 0.475+0.17+0.3 = 0.945

    # 6. Bump assessor_version to 'v2' (preserves v1 and adds new row for replay)
    fake_eval_v2 = LLMRoleSubSignals(
        fit_rationale="Policy v2 assessment: High fit with additional telemetry weight.",
        must_have_coverage=0.8,
        nice_to_have_coverage=0.7,
        location_compatibility=1.0,
        supporting_claim_ids=[str(claim1.id)],
        skill_gaps=[SkillGap(gap="Telemetry", severity="low", blocking=False)],
        company_context={"name": "Platform Co"},
    )
    assess_v2 = RoleAssessor(llm=FakeStructuredLLM(fake_eval_v2), assessor_version="v2").assess(
        role=role, goal=goal, claims=[claim1, claim2]
    )

    upsert_assessments(session, [assess_v2])
    session.commit()

    # Invariant: Total rows in role_assessments is now 2
    final_assessments = session.scalars(
        select(RoleAssessment).order_by(RoleAssessment.assessor_version)
    ).all()
    assert len(final_assessments) == 2

    v1_row = final_assessments[0]
    v2_row = final_assessments[1]

    assert v1_row.assessor_version == "v1"
    assert v2_row.assessor_version == "v2"
    assert v1_row.id != v2_row.id

    # Invariant: Both versions satisfy strict grounding to the user's claims
    assert_assessment_grounding_invariant(session, v1_row, seeded_user.id)
    assert_assessment_grounding_invariant(session, v2_row, seeded_user.id)
