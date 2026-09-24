"""Grounded claim extraction pipeline with strict provenance validation and deduplication."""

from dataclasses import dataclass, field
from uuid import UUID

from pilot.extraction.grounding import GroundingValidator
from pilot.extraction.llm import LLMExtractionError, StructuredLLMClient
from pilot.extraction.schemas import ExtractionBatch
from pilot.ingestion.reader import SourceSpan
from pilot.schemas.evidence import EvidenceClaimCreate, compute_claim_content_hash

SYSTEM_EXTRACTION_PROMPT = """You are Pilot's Grounded Claim Extractor.
Extract atomic, verifiable claims about the candidate from the provided text spans.

CRITICAL RULES:
1. Grounding Guarantee: Every claim must be supported by a verbatim excerpt copied directly from the text.
2. One Fact Per Claim: Keep claims atomic (e.g., 'Deployed vLLM in production' and 'Reduced latency by 45%' are distinct facts).
3. Do NOT Infer: Never assume employers, job titles, dates, seniority levels, degrees, or quantitative metrics that are not explicitly written.
4. Copy Verbatim: The `source_excerpt` must match the source text character-for-character (at least 8 characters).
5. Precision Over Recall: Returning fewer, strictly accurate claims is correct. If a span contains no factual candidate claims, output nothing for it.
"""


@dataclass
class ExtractionDroppedClaim:
    """Record of a proposed claim that was dropped during validation."""

    claim: str
    source_excerpt: str
    reason: str


@dataclass
class ExtractionResult:
    """Consolidated outcome of an extraction run with audit trail for dropped claims."""

    claims: list[EvidenceClaimCreate] = field(default_factory=list)
    dropped: list[ExtractionDroppedClaim] = field(default_factory=list)
    proposed_count: int = 0

    @property
    def grounding_pass_rate(self) -> float:
        """Ratio of successfully verified claims to total proposed claims."""
        if self.proposed_count == 0:
            return 0.0
        return len(self.claims) / self.proposed_count


class GroundedExtractor:
    """Extracts candidate factual claims with strict provenance validation against document spans."""

    def __init__(
        self,
        llm: StructuredLLMClient,
        batch_chars: int = 12000,
        min_confidence: float = 0.0,
    ) -> None:
        self.llm = llm
        self.batch_chars = batch_chars
        self.min_confidence = min_confidence

    def extract(
        self,
        *,
        spans: list[SourceSpan],
        entity_type: str,
        entity_id: UUID,
        source: str,
    ) -> ExtractionResult:
        """Extract atomic evidence claims from spans, strictly validating grounding against source text."""
        result = ExtractionResult()
        if not spans:
            return result

        validator = GroundingValidator(spans)
        batches = self._batch_spans(spans, self.batch_chars)
        seen_hashes: set[str] = set()

        for batch in batches:
            user_prompt = self._build_prompt(batch)
            try:
                extracted_batch = self.llm.complete_structured(
                    system_prompt=SYSTEM_EXTRACTION_PROMPT,
                    user_prompt=user_prompt,
                    schema=ExtractionBatch,
                )
            except LLMExtractionError as err:
                # Log batch drop and continue without killing the ingestion run
                result.dropped.append(
                    ExtractionDroppedClaim(
                        claim="<BATCH_EXTRACTION_FAILED>",
                        source_excerpt="",
                        reason=f"llm_extraction_error: {err}",
                    )
                )
                continue

            for proposed in extracted_batch.claims:
                result.proposed_count += 1

                # 1. Confidence floor
                if proposed.confidence < self.min_confidence:
                    result.dropped.append(
                        ExtractionDroppedClaim(
                            claim=proposed.claim,
                            source_excerpt=proposed.source_excerpt,
                            reason="confidence_below_threshold",
                        )
                    )
                    continue

                # 2. Strict Grounding Gate
                located = validator.locate(proposed.source_excerpt)
                if not located:
                    result.dropped.append(
                        ExtractionDroppedClaim(
                            claim=proposed.claim,
                            source_excerpt=proposed.source_excerpt,
                            reason="ungrounded_source_excerpt",
                        )
                    )
                    continue

                # 3. Provenance Rewriting & In-Run Deduplication
                # Provenance is assigned exclusively from the matched span locator
                rewritten_source_url = f"{located.source_url}#{located.locator}"
                content_hash = compute_claim_content_hash(
                    claim=proposed.claim,
                    source_url=rewritten_source_url,
                )

                if content_hash in seen_hashes:
                    result.dropped.append(
                        ExtractionDroppedClaim(
                            claim=proposed.claim,
                            source_excerpt=proposed.source_excerpt,
                            reason="duplicate_claim",
                        )
                    )
                    continue

                seen_hashes.add(content_hash)

                claim_record = EvidenceClaimCreate(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    claim=proposed.claim.strip(),
                    source=source,
                    source_url=rewritten_source_url,
                    source_excerpt=located.verbatim_text,
                    content_hash=content_hash,
                    confidence=proposed.confidence,
                )
                result.claims.append(claim_record)

        return result

    def _batch_spans(self, spans: list[SourceSpan], batch_chars: int) -> list[list[SourceSpan]]:
        """Partition spans into character-budgeted batches preserving span boundaries."""
        batches: list[list[SourceSpan]] = []
        current_batch: list[SourceSpan] = []
        current_chars = 0

        for span in spans:
            span_len = len(span.text)
            if current_chars + span_len > batch_chars and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(span)
            current_chars += span_len

        if current_batch:
            batches.append(current_batch)
        return batches

    def _build_prompt(self, batch: list[SourceSpan]) -> str:
        """Format document spans into structured user prompt."""
        span_blocks = [
            f"[SPAN {idx + 1}]\n{span.text}\n[/SPAN {idx + 1}]" for idx, span in enumerate(batch)
        ]
        return (
            "Extract all atomic factual claims from the following text spans. "
            "Every claim must cite an exact verbatim excerpt from within these spans:\n\n"
            + "\n\n".join(span_blocks)
        )
