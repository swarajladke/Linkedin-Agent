"""Tests for GroundedExtractor pipeline using scripted FakeLLM implementations."""

import uuid
from typing import TypeVar

from pydantic import BaseModel

from pilot.extraction.extractor import GroundedExtractor
from pilot.extraction.llm import LLMExtractionError, StructuredLLMClient
from pilot.extraction.schemas import ExtractedClaim, ExtractionBatch
from pilot.ingestion.reader import SourceSpan

T = TypeVar("T", bound=BaseModel)


class FakeLLM(StructuredLLMClient):
    """Scripted fake LLM returning preconfigured batches or raising errors."""

    def __init__(self, responses: list[ExtractionBatch | Exception]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_prompts: list[str] = []

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        self.call_count += 1
        self.recorded_prompts.append(user_prompt)
        if not self.responses:
            return schema.model_validate({"claims": []})

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp  # type: ignore


def test_extractor_accepts_grounded_claim_with_rewritten_provenance():
    """Test that a grounded claim is accepted with exact span URL/locator and 64-char hash."""
    span = SourceSpan(
        text="Deployed low-latency inference pipelines using vLLM on Kubernetes.",
        locator="line=10",
        source_url="file:///resume.txt",
    )
    fake_llm = FakeLLM(
        [
            ExtractionBatch(
                claims=[
                    ExtractedClaim(
                        claim="Deployed vLLM on Kubernetes",
                        kind="project",
                        source_excerpt="vLLM on Kubernetes",
                        confidence=0.95,
                    )
                ]
            )
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm)
    entity_id = uuid.uuid4()
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=entity_id,
        source="resume_txt",
    )

    assert len(result.claims) == 1
    assert len(result.dropped) == 0
    assert result.proposed_count == 1
    assert result.grounding_pass_rate == 1.0

    claim = result.claims[0]
    assert claim.claim == "Deployed vLLM on Kubernetes"
    assert claim.source_excerpt == "vLLM on Kubernetes"
    # Provenance must be rewritten from matched span locator
    assert claim.source_url == "file:///resume.txt#line=10;excerpt_chars=47-65"
    assert len(claim.content_hash) == 64
    assert claim.confidence == 0.95


def test_extractor_drops_100_percent_of_hallucinated_claims():
    """Test that claims with fabricated excerpts are 100% dropped."""
    span = SourceSpan(
        text="Built backend systems in Python and Go.",
        locator="line=2",
        source_url="file:///resume.txt",
    )
    fake_llm = FakeLLM(
        [
            ExtractionBatch(
                claims=[
                    ExtractedClaim(
                        claim="Led engineering department of 50 people",
                        kind="experience",
                        source_excerpt="Led engineering department",
                        confidence=0.90,
                    )
                ]
            )
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm)
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_txt",
    )

    assert len(result.claims) == 0
    assert len(result.dropped) == 1
    assert result.dropped[0].reason == "ungrounded_source_excerpt"
    assert result.grounding_pass_rate == 0.0


def test_extractor_mixed_batch_preserves_grounded_and_computes_pass_rate():
    """Test that in a mixed batch, only grounded claims survive and pass rate is accurate."""
    span = SourceSpan(
        text="Architected distributed vector retrieval with pgvector and HNSW indexing.",
        locator="page=1;chars=0-74",
        source_url="file:///resume.pdf",
    )
    fake_llm = FakeLLM(
        [
            ExtractionBatch(
                claims=[
                    ExtractedClaim(
                        claim="Architected distributed vector retrieval",
                        kind="skill",
                        source_excerpt="Architected distributed vector retrieval",
                        confidence=0.92,
                    ),
                    ExtractedClaim(
                        claim="Published research paper at NeurIPS",
                        kind="achievement",
                        source_excerpt="Published research paper",
                        confidence=0.85,
                    ),
                ]
            )
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm)
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_pdf",
    )

    assert len(result.claims) == 1
    assert len(result.dropped) == 1
    assert result.proposed_count == 2
    assert result.grounding_pass_rate == 0.5


def test_extractor_in_run_duplicate_deduplication():
    """Test that duplicate claims with identical content hash within a run collapse to one."""
    span = SourceSpan(
        text="Designed robust ETL pipelines in Apache Airflow.",
        locator="line=8",
        source_url="file:///resume.txt",
    )
    fake_llm = FakeLLM(
        [
            ExtractionBatch(
                claims=[
                    ExtractedClaim(
                        claim="Designed ETL pipelines in Apache Airflow",
                        kind="project",
                        source_excerpt="ETL pipelines in Apache Airflow",
                        confidence=0.90,
                    ),
                    ExtractedClaim(
                        claim="Designed ETL pipelines in Apache Airflow",
                        kind="project",
                        source_excerpt="ETL pipelines in Apache Airflow",
                        confidence=0.92,
                    ),
                ]
            )
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm)
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_txt",
    )

    assert len(result.claims) == 1
    assert len(result.dropped) == 1
    assert result.dropped[0].reason == "duplicate_claim"


def test_extractor_filters_low_confidence():
    """Test that claims below min_confidence are dropped."""
    span = SourceSpan(
        text="Familiar with Docker and basic container operations.",
        locator="line=15",
        source_url="file:///resume.txt",
    )
    fake_llm = FakeLLM(
        [
            ExtractionBatch(
                claims=[
                    ExtractedClaim(
                        claim="Knowledge of Docker",
                        kind="skill",
                        source_excerpt="Docker and basic",
                        confidence=0.40,
                    )
                ]
            )
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm, min_confidence=0.60)
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_txt",
    )

    assert len(result.claims) == 0
    assert len(result.dropped) == 1
    assert result.dropped[0].reason == "confidence_below_threshold"


def test_extractor_llm_failure_does_not_raise():
    """Test that a failed batch is logged as dropped without crashing the entire run."""
    span = SourceSpan(
        text="Valid span text for processing.",
        locator="line=1",
        source_url="file:///resume.txt",
    )
    fake_llm = FakeLLM([LLMExtractionError("Rate limit or context length exceeded")])

    extractor = GroundedExtractor(llm=fake_llm)
    result = extractor.extract(
        spans=[span],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_txt",
    )

    assert len(result.claims) == 0
    assert len(result.dropped) == 1
    assert "llm_extraction_error" in result.dropped[0].reason


def test_extractor_empty_spans_short_circuits():
    """Test that passing an empty span list makes 0 LLM calls."""
    fake_llm = FakeLLM([])
    extractor = GroundedExtractor(llm=fake_llm)

    result = extractor.extract(
        spans=[],
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="resume_txt",
    )

    assert fake_llm.call_count == 0
    assert result.proposed_count == 0
    assert len(result.claims) == 0


def test_extractor_batches_spans_by_budget():
    """Test that spans are correctly partitioned across multiple batches according to char budget."""
    spans = [
        SourceSpan(text="Span 1 text with twenty chars", locator="1", source_url="url"),
        SourceSpan(text="Span 2 text with twenty chars", locator="2", source_url="url"),
        SourceSpan(text="Span 3 text with twenty chars", locator="3", source_url="url"),
    ]
    # Set budget to 35 chars so 3 spans require 2 batches
    fake_llm = FakeLLM(
        [
            ExtractionBatch(claims=[]),
            ExtractionBatch(claims=[]),
        ]
    )

    extractor = GroundedExtractor(llm=fake_llm, batch_chars=35)
    extractor.extract(
        spans=spans,
        entity_type="user",
        entity_id=uuid.uuid4(),
        source="test",
    )

    assert fake_llm.call_count == 2
