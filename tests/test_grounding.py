"""Tests for GroundingValidator asserting exact span location, case/whitespace normalization, and rejection of invalid excerpts."""

from pilot.extraction.grounding import GroundingValidator
from pilot.ingestion.reader import SourceSpan


def test_grounding_exact_match():
    """Test finding an exact substring within a single document span."""
    spans = [
        SourceSpan(
            text="Deployed low-latency inference pipelines using vLLM on Kubernetes.",
            locator="page=1;chars=50-118",
            source_url="file:///resume.pdf",
        )
    ]
    validator = GroundingValidator(spans)

    excerpt = "vLLM on Kubernetes"
    located = validator.locate(excerpt)

    assert located is not None
    assert located.verbatim_text == "vLLM on Kubernetes"
    assert located.source_url == "file:///resume.pdf"
    assert located.locator == "page=1;chars=50-118;excerpt_chars=47-65"
    assert validator.is_grounded(excerpt) is True


def test_grounding_whitespace_and_case_tolerance_returns_original_formatting():
    """Test that query whitespace and casing differences still return the original formatting verbatim."""
    spans = [
        SourceSpan(
            text="  Spearheaded   Multi-Agent  Orchestration   Framework!  ",
            locator="line=12-14",
            source_url="file:///resume.txt",
        )
    ]
    validator = GroundingValidator(spans)

    # Lowercase with single spaces
    excerpt = "multi-agent orchestration framework!"
    located = validator.locate(excerpt)

    assert located is not None
    # Must recover original capitalization and internal spacing verbatim
    assert located.verbatim_text == "Multi-Agent  Orchestration   Framework!"
    assert located.locator == "line=12-14;excerpt_chars=16-55"


def test_grounding_rejects_hallucination():
    """Test that text not present in any span returns None."""
    spans = [
        SourceSpan(
            text="Senior Machine Learning Engineer at TechCorp Inc.",
            locator="line=5",
            source_url="file:///resume.txt",
        )
    ]
    validator = GroundingValidator(spans)

    assert validator.locate("Principal AI Architect at Google") is None
    assert validator.is_grounded("Principal AI Architect at Google") is False


def test_grounding_rejects_paraphrase():
    """Test that a semantic paraphrase lacking exact text is rejected."""
    spans = [
        SourceSpan(
            text="Built retrieval-augmented generation pipeline using pgvector.",
            locator="line=20",
            source_url="file:///resume.txt",
        )
    ]
    validator = GroundingValidator(spans)

    # Paraphrase expressing the same meaning but different words
    paraphrase = "created a retrieval augmented generation service"
    assert validator.locate(paraphrase) is None
    assert validator.is_grounded(paraphrase) is False


def test_grounding_rejects_cross_span_splice():
    """Test that an excerpt spliced across two separate spans is rejected."""
    spans = [
        SourceSpan(
            text="Designed high throughput inference server.",
            locator="line=10",
            source_url="file:///resume.txt",
        ),
        SourceSpan(
            text="Achieved 45 percent latency reduction.",
            locator="line=11",
            source_url="file:///resume.txt",
        ),
    ]
    validator = GroundingValidator(spans)

    # Spliced across span 1 and span 2
    spliced = "inference server. Achieved 45 percent"
    assert validator.locate(spliced) is None
    assert validator.is_grounded(spliced) is False


def test_grounding_rejects_too_short_excerpt():
    """Test that an excerpt shorter than min_excerpt_chars is rejected."""
    spans = [
        SourceSpan(
            text="Experienced Python and Rust software developer.",
            locator="line=1",
            source_url="file:///resume.txt",
        )
    ]
    validator = GroundingValidator(spans, min_excerpt_chars=8)

    # 4 chars < 8
    assert validator.locate("Rust") is None
    assert validator.is_grounded("Rust") is False
