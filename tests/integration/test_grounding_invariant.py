"""Adversarial integration tests verifying that ungrounded, hallucinated, spliced, or mutated claims are 100% dropped."""

from typing import TypeVar

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import EvidenceClaim, User
from pilot.extraction.extractor import GroundedExtractor
from pilot.extraction.grounding import GroundingValidator
from pilot.extraction.llm import StructuredLLMClient
from pilot.extraction.repository import upsert_evidence_claims
from pilot.extraction.schemas import ExtractedClaim, ExtractionBatch
from pilot.ingestion.reader import SourceSpan
from tests.integration.conftest import assert_grounding_invariant

pytestmark = pytest.mark.integration

T = TypeVar("T", bound=BaseModel)

SPAN_A = SourceSpan(
    text="Designed low-latency inference pipelines using vLLM and TensorRT-LLM, reducing latency by 45%.",
    locator="line=10",
    source_url="file:///resume.txt",
)
SPAN_B = SourceSpan(
    text="Built automated evaluation harness measuring exact-match and semantic accuracy across 2,000 requests.",
    locator="line=11",
    source_url="file:///resume.txt",
)
SPAN_C = SourceSpan(
    text="Alex's automated test harness verified 100% of candidate's claims.",
    locator="line=12",
    source_url="file:///resume.txt",
)
KNOWN_SPANS = [SPAN_A, SPAN_B, SPAN_C]


class AdversarialLLM(StructuredLLMClient):
    """Fake LLM returning adversarial hallucinated, spliced, modified, and look-alike claims."""

    def __init__(self, batch: ExtractionBatch) -> None:
        self.batch = batch
        self.call_count = 0

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.call_count += 1
        return self.batch  # type: ignore


@pytest.fixture
def adversarial_batch() -> ExtractionBatch:
    """Construct an adversarial batch containing 1 valid claim and 9 crafted invalid claims."""
    return ExtractionBatch(
        claims=[
            # 1. Verbatim quote with different casing and collapsed/expanded whitespace -> MUST GROUND
            ExtractedClaim.model_construct(
                claim="Deployed low-latency inference pipelines using vLLM and TensorRT-LLM",
                kind="project",
                source_excerpt="  designed   LOW-LATENCY   inference   pipelines   using   vllm  ",
                confidence=0.95,
            ),
            # 2. Plausible paraphrase of a real span -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Engineered high throughput model serving with vLLM",
                kind="project",
                source_excerpt="Engineered high throughput model serving with vLLM",
                confidence=0.95,
            ),
            # 3. Real span with one word silently changed (2,000 -> 20,000) -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Semantic accuracy across 20,000 requests",
                kind="achievement",
                source_excerpt="semantic accuracy across 20,000 requests",
                confidence=0.90,
            ),
            # 4. Text spliced from the tail of span A and head of span B -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Reduced latency and built eval harness",
                kind="experience",
                source_excerpt="reducing latency by 45%. Built automated evaluation harness",
                confidence=0.88,
            ),
            # 5. Excerpt that is a real span plus one fabricated trailing clause -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Inference pipelines with guaranteed uptime SLA",
                kind="project",
                source_excerpt="low-latency inference pipelines using vLLM and TensorRT-LLM with guaranteed 99.9% uptime SLA",
                confidence=0.91,
            ),
            # 6. Single common word genuinely appearing in a span (below min length of 8) -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Uses modern inference frameworks",
                kind="skill",
                source_excerpt="using",
                confidence=0.99,
            ),
            # 7. Empty string excerpt -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Empty excerpt proposed claim",
                kind="skill",
                source_excerpt="",
                confidence=0.90,
            ),
            # 8. Whitespace-only excerpt -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Whitespace only excerpt proposed claim",
                kind="skill",
                source_excerpt="   \t\n  ",
                confidence=0.90,
            ),
            # 9. Unicode look-alike: en-dash \u2013 instead of ASCII hyphen - -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Exact match accuracy benchmark",
                kind="achievement",
                source_excerpt="exact–match",
                confidence=0.95,
            ),
            # 10. Unicode look-alike: curly apostrophe \u2019 instead of ASCII straight ' -> MUST DROP
            ExtractedClaim.model_construct(
                claim="Verified candidate claims",
                kind="achievement",
                source_excerpt="candidate’s claims",
                confidence=0.95,
            ),
        ]
    )


def test_adversarial_extraction_hallucination_drop_rate_is_exact_one(
    session: Session,
    seeded_user: User,
    adversarial_batch: ExtractionBatch,
):
    """Invariant: The extractor must achieve an exact 1.0 drop rate on all hallucinated, mutated, or ungrounded claims, ensuring only verifiable verbatim excerpts enter evidence_claims."""
    fake_llm = AdversarialLLM(adversarial_batch)
    extractor = GroundedExtractor(llm=fake_llm)

    result = extractor.extract(
        spans=KNOWN_SPANS,
        entity_type="user",
        entity_id=seeded_user.id,
        source="resume",
    )

    # Invariant: exactly 1 claim grounded, exactly 9 adversarial claims dropped
    assert result.proposed_count == 10
    assert len(result.claims) == 1
    assert len(result.dropped) == 9

    # Check reason on all dropped claims
    for dropped in result.dropped:
        assert dropped.reason == "ungrounded_source_excerpt"

    # Hallucination drop rate invariant: 9 ungrounded proposals / 9 ungrounded proposals == 1.0
    hallucinations_proposed = 9
    hallucinations_dropped = len(
        [d for d in result.dropped if d.reason == "ungrounded_source_excerpt"]
    )
    assert hallucinations_dropped / hallucinations_proposed == 1.0

    # Persist the single accepted claim and assert aggregate grounding invariant
    upserted = upsert_evidence_claims(session, result.claims)
    session.commit()
    assert upserted == 1

    persisted = session.scalars(
        select(EvidenceClaim).where(EvidenceClaim.entity_id == seeded_user.id)
    ).all()
    assert len(persisted) == 1

    assert_grounding_invariant(persisted, KNOWN_SPANS)


def test_grounding_validator_rejects_paraphrase():
    """Invariant: Semantic paraphrases lacking exact character-level verbatim text must be rejected to prevent interpretive drift."""
    validator = GroundingValidator(KNOWN_SPANS)
    assert validator.locate("Engineered high throughput model serving with vLLM") is None


def test_grounding_validator_rejects_silent_token_mutation():
    """Invariant: Subtle mutations to metrics, numbers, or terms (e.g. 2,000 -> 20,000) must be rejected to preserve numerical integrity."""
    validator = GroundingValidator(KNOWN_SPANS)
    assert validator.locate("semantic accuracy across 20,000 requests") is None


def test_grounding_validator_rejects_cross_span_splice():
    """Invariant: Excerpts stitched across separate, disjoint document spans must be rejected to maintain span-level provenance boundaries."""
    validator = GroundingValidator(KNOWN_SPANS)
    assert validator.locate("reducing latency by 45%. Built automated evaluation harness") is None


def test_grounding_validator_rejects_fabricated_trailing_clause():
    """Invariant: Real spans augmented with ungrounded trailing clauses must be rejected in their entirety rather than partially trusted."""
    validator = GroundingValidator(KNOWN_SPANS)
    assert (
        validator.locate(
            "low-latency inference pipelines using vLLM and TensorRT-LLM with guaranteed 99.9% uptime SLA"
        )
        is None
    )


def test_grounding_validator_rejects_below_minimum_length():
    """Invariant: Excerpts below the minimum character threshold must be rejected to prevent spurious matching of trivial common words."""
    validator = GroundingValidator(KNOWN_SPANS, min_excerpt_chars=8)
    assert validator.locate("using") is None


def test_grounding_validator_rejects_empty_and_whitespace_without_exception():
    """Invariant: Empty or whitespace-only queries must fail fast and return None without raising unhandled runtime exceptions."""
    validator = GroundingValidator(KNOWN_SPANS)
    assert validator.locate("") is None
    assert validator.locate("   \t\n  ") is None


def test_grounding_validator_rejects_unicode_homoglyphs_and_lookalikes():
    """Invariant: The validator must normalize whitespace and ASCII casing only; unicode look-alikes like en-dashes and curly apostrophes must be rejected."""
    validator = GroundingValidator(KNOWN_SPANS)
    # Span has ASCII hyphen 'exact-match', query has unicode en-dash 'exact–match' (\u2013)
    assert validator.locate("exact–match") is None
    # Span has ASCII straight quote "candidate's", query has unicode curly quote "candidate’s" (\u2019)
    assert validator.locate("candidate’s claims") is None


def test_grounding_validator_accepts_verbatim_with_case_and_whitespace_variation():
    """Invariant: Legitimate verbatim excerpts must successfully resolve regardless of variations in capitalization or extraneous whitespace."""
    validator = GroundingValidator(KNOWN_SPANS)
    located = validator.locate("  designed   LOW-LATENCY   inference   pipelines   using   vllm  ")
    assert located is not None
    assert located.verbatim_text == "Designed low-latency inference pipelines using vLLM"
    assert located.source_url == "file:///resume.txt"
    assert located.span == SPAN_A
