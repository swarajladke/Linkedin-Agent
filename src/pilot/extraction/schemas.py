"""Pydantic schemas for structured LLM claim extraction."""

from typing import Literal

from pydantic import BaseModel, Field

ClaimKind = Literal[
    "skill",
    "project",
    "experience",
    "education",
    "achievement",
    "other",
]


class ExtractedClaim(BaseModel):
    """An atomic candidate claim extracted by an LLM with required source excerpt grounding."""

    claim: str = Field(
        min_length=5,
        description="Factual, atomic statement about the candidate. One distinct fact only.",
    )
    kind: ClaimKind = Field(
        description="Semantic category of the claim.",
    )
    source_excerpt: str = Field(
        min_length=8,
        description="Verbatim character-for-character excerpt copied directly from the source text.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in the veracity and clarity of the extracted claim.",
    )


class ExtractionBatch(BaseModel):
    """Batch container for claims extracted from a collection of document spans."""

    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="List of extracted claims found in the provided text.",
    )
