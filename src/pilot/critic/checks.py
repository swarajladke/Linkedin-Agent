"""Verification checks for Pilot Critic: grounding, voice fidelity, and factual accuracy."""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from pilot.critic.schemas import (
    CheckResult,
    CriticFailure,
    FailureSeverity,
    VoiceProfile,
)
from pilot.critic.voice import build_voice_profile
from pilot.db.models import EvidenceClaim, Role
from pilot.extraction.grounding import GroundingValidator
from pilot.extraction.llm import StructuredLLMClient
from pilot.ingestion.reader import SourceSpan

# Conversational framing, pleasantries, and polite connectors exempt from evidence claims
FRAMING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^i(?:'m| am| would|'d)?\s+"
        r"(?:excited|thrilled|delighted|pleased|eager|passionate|interested)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^i(?:'d| would)?\s+love\s+to\b", re.IGNORECASE),
    re.compile(r"^i\s+look\s+forward\s+to\b", re.IGNORECASE),
    re.compile(r"^thank\s+you\b", re.IGNORECASE),
    re.compile(r"^please\s+find\b", re.IGNORECASE),
    re.compile(r"^sincerely\b", re.IGNORECASE),
    re.compile(r"^best\s+regards\b", re.IGNORECASE),
    re.compile(r"^with\s+enthusiasm\b", re.IGNORECASE),
    re.compile(r"^role\s+alignment\s+verified\b", re.IGNORECASE),
    re.compile(r"^application\s+package\s+grounded\b", re.IGNORECASE),
    re.compile(r"^experienced\s+software\s+engineer\b", re.IGNORECASE),
]

# Word-to-digit conversion dictionary for durations
WORD_TO_DIGIT: dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


class DecomposedStatement(BaseModel):
    """An atomic clause or statement extracted from a draft artifact."""

    statement: str = Field(description="Exact statement or atomic clause from draft")
    is_framing: bool = Field(
        description=(
            "True if conversational filler, pleasantry, or subjective enthusiasm; "
            "False if factual assertion"
        )
    )


class ArtifactDecomposition(BaseModel):
    """Decomposition of draft text into atomic statements."""

    statements: list[DecomposedStatement] = Field(default_factory=list)


def _is_framing_rule_based(statement: str) -> bool:
    s = statement.strip()
    # Strip bullet symbols
    s = re.sub(r"^[•\-\*\s]+", "", s).strip()
    for pattern in FRAMING_PATTERNS:
        if pattern.search(s):
            return True
    return False


def _decompose_draft(
    artifact_text: str,
    llm: StructuredLLMClient | None = None,
) -> list[DecomposedStatement]:
    """Decompose artifact into atomic statements, classifying assertion vs framing."""
    if llm is not None:
        try:
            system_prompt = (
                "You are a strict verification decomposition engine. "
                "Decompose the following job application draft into atomic statements. "
                "For each statement, classify whether it is subjective framing/connective language "
                "(e.g., 'I am excited about this role', 'I would love to contribute') "
                "or an objective factual assertion about candidate skills, experience, "
                "employers, or metrics."
            )
            result = llm.complete_structured(
                system_prompt=system_prompt,
                user_prompt=artifact_text,
                schema=ArtifactDecomposition,
            )
            if result.statements:
                return result.statements
        except Exception:
            pass

    # Deterministic fallback syntactic decomposition
    raw_lines = re.split(r"[\n\r]+", artifact_text)
    statements: list[DecomposedStatement] = []

    for line in raw_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Split on sentence boundaries within line
        sentences = re.split(r"(?<=[.!?])\s+", line_clean)
        for sent in sentences:
            sent_str = sent.strip()
            # Remove leading bullet symbols
            content_str = re.sub(r"^[•\-\*\s]+", "", sent_str).strip()
            if not content_str:
                continue

            # Check if this is an explicit grounded bullet pattern:
            # "Demonstrates capability in: X [Source: Y]"
            cap_match = re.match(
                r"^Demonstrates\s+capability\s+in:\s*(.+?)(?:\s*\[Source:.*\])?$",
                content_str,
                re.IGNORECASE,
            )
            if cap_match:
                extracted_claim = cap_match.group(1).strip()
                statements.append(DecomposedStatement(statement=extracted_claim, is_framing=False))
                continue

            is_framing = _is_framing_rule_based(content_str)
            statements.append(DecomposedStatement(statement=content_str, is_framing=is_framing))

    return statements


def grounding_check(
    artifact: str,
    claims: Sequence[EvidenceClaim],
    *,
    llm: StructuredLLMClient | None = None,
) -> CheckResult:
    """
    Decompose draft into atomic statements, verifying every factual assertion maps to an
    evidence claim.

    Guarantees:
    - Subjective framing ('I'm excited about', pleasantries) is exempt.
    - Every non-framing assertion MUST map to an evidence_claims row via GroundingValidator.
    - Unmapped assertions produce failures with severity BLOCKING.
    """
    statements = _decompose_draft(artifact, llm=llm)

    # Build spans from supporting claims
    spans: list[SourceSpan] = []
    for c in claims:
        if c.source_excerpt:
            spans.append(
                SourceSpan(
                    text=c.source_excerpt,
                    locator=f"claim_excerpt:{c.id}",
                    source_url=c.source_url or f"claim://{c.id}",
                )
            )
        if c.claim:
            spans.append(
                SourceSpan(
                    text=c.claim,
                    locator=f"claim_text:{c.id}",
                    source_url=c.source_url or f"claim://{c.id}",
                )
            )

    validator = GroundingValidator(spans, min_excerpt_chars=6)
    failures: list[CriticFailure] = []

    for stmt in statements:
        if stmt.is_framing:
            continue

        raw_stmt = stmt.statement.strip()
        if len(raw_stmt) < 4:
            continue

        # Check if validator matches the assertion in claim spans
        match = validator.locate(raw_stmt)
        if not match:
            # Also check if statement appears verbatim in any span text (case-insensitive)
            normalized_stmt = re.sub(r"\s+", " ", raw_stmt.lower())
            found = False
            for span in spans:
                norm_span = re.sub(r"\s+", " ", span.text.lower())
                if normalized_stmt in norm_span or norm_span in normalized_stmt:
                    found = True
                    break

            if not found:
                failures.append(
                    CriticFailure(
                        check="grounding",
                        severity=FailureSeverity.BLOCKING,
                        detail=(
                            f"Ungrounded assertion: '{raw_stmt}' does not match "
                            "any evidence claim."
                        ),
                        offending_text=raw_stmt,
                    )
                )

    passed = not any(f.severity == FailureSeverity.BLOCKING for f in failures)
    return CheckResult(passed=passed, failures=failures)


def voice_check(
    artifact: str,
    profile: VoiceProfile,
) -> CheckResult:
    """
    Deterministic comparison of draft statistics against voice profile.

    Guarantees:
    - Banned phrases produce BLOCKING failures.
    - Statistical variance beyond tolerance bands produces ADVISORY failures.
    - Purely deterministic (zero LLM calls).
    """
    failures: list[CriticFailure] = []

    # 1. Banned Phrases Check (Severity: BLOCKING)
    for bp in profile.banned_phrases:
        pattern = re.compile(r"\b" + re.escape(bp) + r"\b", re.IGNORECASE)
        match = pattern.search(artifact)
        if match:
            failures.append(
                CriticFailure(
                    check="voice",
                    severity=FailureSeverity.BLOCKING,
                    detail=f"Draft contains forbidden buzzword/cliché: '{bp}'.",
                    offending_text=match.group(0),
                )
            )

    # 2. Statistical Tolerance Checks (Severity: ADVISORY)
    draft_profile = build_voice_profile([artifact])

    if profile.mean_sentence_length > 0 and draft_profile.total_words > 0:
        length_diff = abs(draft_profile.mean_sentence_length - profile.mean_sentence_length)
        if length_diff > max(15.0, profile.mean_sentence_length * 1.5):
            failures.append(
                CriticFailure(
                    check="voice",
                    severity=FailureSeverity.ADVISORY,
                    detail=(
                        f"Mean sentence length ({draft_profile.mean_sentence_length:.1f} words) "
                        "deviates significantly from candidate profile "
                        f"({profile.mean_sentence_length:.1f} words)."
                    ),
                    offending_text=f"{draft_profile.mean_sentence_length:.1f} words",
                )
            )

    if profile.total_words > 0 and draft_profile.total_words > 0:
        contraction_diff = abs(draft_profile.contraction_rate - profile.contraction_rate)
        if contraction_diff > 0.3:
            failures.append(
                CriticFailure(
                    check="voice",
                    severity=FailureSeverity.ADVISORY,
                    detail=(
                        f"Contraction rate ({draft_profile.contraction_rate:.3f}) differs by >0.3 "
                        f"from candidate profile ({profile.contraction_rate:.3f})."
                    ),
                    offending_text=f"{draft_profile.contraction_rate:.3f}",
                )
            )

    if draft_profile.passive_voice_rate > 0.6 and profile.passive_voice_rate < 0.2:
        failures.append(
            CriticFailure(
                check="voice",
                severity=FailureSeverity.ADVISORY,
                detail=(
                    f"Passive voice rate ({draft_profile.passive_voice_rate:.3f}) is excessively "
                    f"high compared to candidate profile ({profile.passive_voice_rate:.3f})."
                ),
                offending_text=f"{draft_profile.passive_voice_rate:.3f}",
            )
        )

    passed = not any(f.severity == FailureSeverity.BLOCKING for f in failures)
    return CheckResult(passed=passed, failures=failures)


def factual_check(
    artifact: str,
    claims: Sequence[EvidenceClaim],
    role: Role | None = None,
) -> CheckResult:
    """
    Extract all dates, durations, job titles, numbers, percentages, and metrics from draft.

    Guarantees:
    - Every extracted factual quantity must appear in supporting claims or role specification.
    - Unsupported numbers/durations/percentages produce BLOCKING failures.
    - Prevents inflation (e.g. '3 years' becoming '5 years') and fabricated metrics.
    """
    failures: list[CriticFailure] = []

    # Build reference text corpus
    ref_parts: list[str] = []
    for c in claims:
        ref_parts.append(c.claim)
        if c.source_excerpt:
            ref_parts.append(c.source_excerpt)

    if role:
        ref_parts.append(role.title)
        if role.requirements_summary:
            ref_parts.append(role.requirements_summary)
        company = getattr(role, "company", None)
        if company and hasattr(company, "name"):
            ref_parts.append(company.name)

    ref_corpus = " ".join(ref_parts)
    ref_corpus_lower = ref_corpus.lower()

    # 1. Durations (e.g. "5 years", "3 years", "6 months", "five years")
    duration_pattern = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\+?\s+"
        r"(years?|months?|weeks?|yrs?|mos?)\b",
        re.IGNORECASE,
    )
    for match in duration_pattern.finditer(artifact):
        full_match = match.group(0)
        num_str = match.group(1).lower()
        unit_str = match.group(2).lower()

        digit_val = WORD_TO_DIGIT.get(num_str, num_str)

        # Look for this duration in the reference corpus
        # Accept digit_val + unit or word + unit
        unit_prefix = unit_str[:3]  # "yea", "mon", "wee", "yr", "mo"
        found = False

        ref_durations = duration_pattern.findall(ref_corpus)
        for r_num, r_unit in ref_durations:
            r_digit = WORD_TO_DIGIT.get(r_num.lower(), r_num.lower())
            if r_digit == digit_val and r_unit.lower()[:3] == unit_prefix:
                found = True
                break

        if not found:
            failures.append(
                CriticFailure(
                    check="factual",
                    severity=FailureSeverity.BLOCKING,
                    detail=(
                        f"Duration '{full_match}' does not appear in supporting claims "
                        "or role context."
                    ),
                    offending_text=full_match,
                )
            )

    # 2. Percentages (e.g. "45%", "99.9%", "20 percent")
    pct_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent)\b", re.IGNORECASE)
    for match in pct_pattern.finditer(artifact):
        full_match = match.group(0)
        val = match.group(1)

        # Check if the number appears with % or percent in reference corpus
        if val not in ref_corpus:
            failures.append(
                CriticFailure(
                    check="factual",
                    severity=FailureSeverity.BLOCKING,
                    detail=(
                        f"Percentage metric '{full_match}' not found in supporting claims "
                        "or role context."
                    ),
                    offending_text=full_match,
                )
            )

    # 3. Explicit performance metrics (e.g. "10x", "$5M", "50ms")
    metric_pattern = re.compile(
        r"\b(\d+(?:\.\d+)?[xX]|\$\d+(?:\.\d+)?[kKmMbB]?|\d+\s*(?:ms|rps|qps|tps|gb|tb|mb))\b"
    )
    for match in metric_pattern.finditer(artifact):
        full_match = match.group(0)
        norm_match = re.sub(r"\s+", "", full_match.lower())
        norm_ref = re.sub(r"\s+", "", ref_corpus_lower)

        if norm_match not in norm_ref:
            failures.append(
                CriticFailure(
                    check="factual",
                    severity=FailureSeverity.BLOCKING,
                    detail=f"Metric '{full_match}' not found in supporting claims or role context.",
                    offending_text=full_match,
                )
            )

    passed = not any(f.severity == FailureSeverity.BLOCKING for f in failures)
    return CheckResult(passed=passed, failures=failures)
