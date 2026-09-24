"""Database persistence and idempotent upsert repository for evidence claims."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from pilot.db.models import EvidenceClaim
from pilot.schemas.evidence import EvidenceClaimCreate


def upsert_evidence_claims(session: Session, claims: list[EvidenceClaimCreate]) -> int:
    """Idempotently insert or update evidence claims on (entity_id, content_hash) conflict.

    Refreshes confidence, source_excerpt, and verified_at timestamp upon conflict.
    Returns the number of affected rows.
    """
    if not claims:
        return 0

    values = [
        {
            "entity_type": claim.entity_type,
            "entity_id": claim.entity_id,
            "claim": claim.claim,
            "source": claim.source,
            "source_url": claim.source_url,
            "source_excerpt": claim.source_excerpt,
            "content_hash": claim.content_hash,
            "confidence": claim.confidence,
        }
        for claim in claims
    ]

    stmt = insert(EvidenceClaim).values(values)
    upsert_stmt = stmt.on_conflict_do_update(
        constraint="uq_evidence_claims_entity_hash",
        set_={
            "confidence": stmt.excluded.confidence,
            "source_excerpt": stmt.excluded.source_excerpt,
            "verified_at": func.now(),
        },
    )

    result = session.execute(upsert_stmt)
    session.flush()
    return result.rowcount
