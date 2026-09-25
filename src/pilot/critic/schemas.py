"""Pydantic schemas for the Critic gate, statistical voice profiling, and verification checks."""

import enum
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Statistical Voice Profile Schemas
# ============================================================================
class VoiceProfile(BaseModel):
    """Statistical voice profile computed across candidate writing samples.

    All metrics are computed deterministically without opaque LLM prose.
    """

    mean_sentence_length: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean sentence length in words",
    )
    median_sentence_length: float = Field(
        default=0.0,
        ge=0.0,
        description="Median sentence length in words",
    )
    sentence_length_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Sample variance of sentence lengths in words",
    )
    mean_paragraph_length: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean paragraph length in words",
    )
    contraction_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Number of contractions divided by total words",
    )
    first_person_pronoun_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="First-person pronouns ('I', 'me', 'my', 'we', 'our', etc.) per word",
    )
    passive_voice_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Passive voice constructions divided by total sentences",
    )
    hedging_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Hedging words ('maybe', 'perhaps', 'seem', etc.) per word",
    )
    exclamation_freq: float = Field(
        default=0.0,
        ge=0.0,
        description="Exclamation points count per 100 words",
    )
    em_dash_freq: float = Field(
        default=0.0,
        ge=0.0,
        description="Em-dashes count per 100 words",
    )
    semicolon_freq: float = Field(
        default=0.0,
        ge=0.0,
        description="Semicolons count per 100 words",
    )
    type_token_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Unique words divided by total words (lexical diversity)",
    )
    common_sentence_openers: list[str] = Field(
        default_factory=list,
        description="Top sentence starting words or phrases",
    )
    banned_phrases: list[str] = Field(
        default_factory=list,
        description="Forbidden clichés and buzzwords never used by the candidate",
    )
    sample_count: int = Field(
        default=1,
        ge=0,
        description="Number of writing samples included in this profile",
    )
    total_words: int = Field(
        default=0,
        ge=0,
        description="Total words analyzed across all writing samples",
    )


# ============================================================================
# Check & Review Result Schemas
# ============================================================================
class FailureSeverity(enum.StrEnum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


class CriticFailure(BaseModel):
    """Specific defect identified by a check."""

    check: str = Field(description="Check name: 'grounding' | 'voice' | 'factual'")
    severity: FailureSeverity = Field(description="Severity: 'blocking' or 'advisory'")
    detail: str = Field(description="Detailed explanation of the failure")
    offending_text: str = Field(
        description="Specific substring in the artifact that triggered failure"
    )


class CheckResult(BaseModel):
    """Result of an individual critic verification check."""

    passed: bool
    failures: list[CriticFailure] = Field(default_factory=list)


class CriticVerdict(enum.StrEnum):
    PASS = "pass"
    REGENERATE = "regenerate"
    DROP = "drop"


class CriticReviewRecord(BaseModel):
    """Structured review record representing a single critic evaluation attempt."""

    id: UUID | None = None
    action_id: UUID
    attempt: int
    artifact_text: str
    grounding_passed: bool
    voice_passed: bool
    factual_passed: bool
    verdict: CriticVerdict
    failures: list[CriticFailure] = Field(default_factory=list)
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
