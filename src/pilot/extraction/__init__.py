"""Extraction module: structured LLM extraction, grounding validation, span flattening, and evidence claims repository."""

from pilot.extraction.extractor import (
    ExtractionDroppedClaim,
    ExtractionResult,
    GroundedExtractor,
)
from pilot.extraction.grounding import GroundedExcerpt, GroundingValidator
from pilot.extraction.llm import LLMExtractionError, OpenAIStructuredClient, StructuredLLMClient
from pilot.extraction.repository import upsert_evidence_claims
from pilot.extraction.schemas import ExtractedClaim, ExtractionBatch
from pilot.extraction.spans import github_to_spans

__all__ = [
    "ExtractedClaim",
    "ExtractionBatch",
    "StructuredLLMClient",
    "OpenAIStructuredClient",
    "LLMExtractionError",
    "GroundedExcerpt",
    "GroundingValidator",
    "github_to_spans",
    "GroundedExtractor",
    "ExtractionResult",
    "ExtractionDroppedClaim",
    "upsert_evidence_claims",
]
