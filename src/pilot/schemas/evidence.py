"""Pydantic schemas for Evidence Claims with provenance hashing and confidence validation."""

import hashlib
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


def compute_claim_content_hash(claim: str, source_url: str) -> str:
    """Compute deterministic SHA-256 hash of normalized claim text and source URL."""
    normalized = f"{claim.strip().lower()}|{source_url.strip()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class EvidenceClaimBase(BaseModel):
    entity_type: str = Field(
        min_length=1,
        max_length=50,
        description="Discriminator: user | company | person | role",
    )
    entity_id: UUID
    claim: str = Field(min_length=5, description="Verifiable atomic claim statement")
    source: str = Field(min_length=1, max_length=100)
    source_url: str = Field(min_length=1, description="Source locator / commit / doc path")
    source_excerpt: str = Field(min_length=1, description="Literal text span from source document")
    content_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash for deduplication",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def populate_content_hash(self) -> "EvidenceClaimBase":
        """Compute content hash automatically if omitted."""
        if not self.content_hash:
            self.content_hash = compute_claim_content_hash(self.claim, self.source_url)
        return self


class EvidenceClaimCreate(EvidenceClaimBase):
    pass


class EvidenceClaimRead(EvidenceClaimBase):
    id: UUID
    verified_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
