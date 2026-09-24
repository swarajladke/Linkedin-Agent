"""Integration tests for evidence_claims upsert, content hash determinism, and deduplication."""

import time
import uuid

from sqlalchemy import select

from pilot.db.models import EvidenceClaim
from pilot.extraction.repository import upsert_evidence_claims
from pilot.schemas.evidence import EvidenceClaimCreate, compute_claim_content_hash


def test_content_hash_determinism_and_normalization():
    """Test that content_hash produces identical SHA-256 for case/whitespace-equivalent claim and URL."""
    hash1 = compute_claim_content_hash(
        claim="  Deployed vLLM inference server  ",
        source_url="file:///resume.pdf#line=12",
    )
    hash2 = compute_claim_content_hash(
        claim="deployed vllm inference server",
        source_url="file:///resume.pdf#line=12",
    )
    assert len(hash1) == 64
    assert hash1 == hash2


def test_upsert_evidence_claims_idempotent_and_refreshes_confidence(db_session):
    """Test that upserting the exact same claim twice updates confidence and leaves 1 row."""
    entity_id = uuid.uuid4()
    claim_text = "Designed multi-agent evaluation framework"
    source_url = "file:///resume.txt#line=20;excerpt_chars=0-40"
    content_hash = compute_claim_content_hash(claim_text, source_url)

    first_claim = EvidenceClaimCreate(
        entity_type="user",
        entity_id=entity_id,
        claim=claim_text,
        source="resume_txt",
        source_url=source_url,
        source_excerpt="Designed multi-agent evaluation framework",
        content_hash=content_hash,
        confidence=0.80,
    )

    # First insert
    count1 = upsert_evidence_claims(db_session, [first_claim])
    assert count1 == 1

    row1 = db_session.scalar(
        select(EvidenceClaim).where(
            EvidenceClaim.entity_id == entity_id,
            EvidenceClaim.content_hash == content_hash,
        )
    )
    assert row1 is not None
    assert row1.confidence == 0.80
    verified_at_1 = row1.verified_at

    # Small pause to ensure timestamp increments on update
    time.sleep(0.05)

    # Second upsert with updated confidence
    second_claim = EvidenceClaimCreate(
        entity_type="user",
        entity_id=entity_id,
        claim=claim_text,
        source="resume_txt",
        source_url=source_url,
        source_excerpt="Designed multi-agent evaluation framework",
        content_hash=content_hash,
        confidence=0.95,
    )
    count2 = upsert_evidence_claims(db_session, [second_claim])
    assert count2 == 1

    # Query all rows for this entity
    rows = db_session.scalars(
        select(EvidenceClaim).where(
            EvidenceClaim.entity_id == entity_id,
            EvidenceClaim.content_hash == content_hash,
        )
    ).all()

    # Must still be exactly 1 row
    assert len(rows) == 1
    assert rows[0].confidence == 0.95
    assert rows[0].verified_at >= verified_at_1


def test_same_claim_for_different_entities_keeps_both(db_session):
    """Test that unique constraint (entity_id, content_hash) allows identical claims for distinct entities."""
    user_1 = uuid.uuid4()
    user_2 = uuid.uuid4()
    claim_text = "Experienced in PyTorch and CUDA kernel optimization"
    source_url = "https://github.com/repo#readme"
    content_hash = compute_claim_content_hash(claim_text, source_url)

    claim_user_1 = EvidenceClaimCreate(
        entity_type="user",
        entity_id=user_1,
        claim=claim_text,
        source="github_repo",
        source_url=source_url,
        source_excerpt="PyTorch and CUDA kernel optimization",
        content_hash=content_hash,
        confidence=0.90,
    )
    claim_user_2 = EvidenceClaimCreate(
        entity_type="user",
        entity_id=user_2,
        claim=claim_text,
        source="github_repo",
        source_url=source_url,
        source_excerpt="PyTorch and CUDA kernel optimization",
        content_hash=content_hash,
        confidence=0.92,
    )

    upsert_evidence_claims(db_session, [claim_user_1, claim_user_2])

    row_1 = db_session.scalar(select(EvidenceClaim).where(EvidenceClaim.entity_id == user_1))
    row_2 = db_session.scalar(select(EvidenceClaim).where(EvidenceClaim.entity_id == user_2))

    assert row_1 is not None
    assert row_2 is not None
    assert row_1.content_hash == row_2.content_hash
    assert row_1.entity_id != row_2.entity_id
