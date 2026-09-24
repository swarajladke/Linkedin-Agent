"""Role assessment engine calculating grounded fit scores, skill gaps, and recommended actions."""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from pilot.db.models import EvidenceClaim, Goal, Role
from pilot.extraction.llm import StructuredLLMClient
from pilot.schemas.intelligence import RoleAssessmentCreate, SkillGap


class LLMRoleSubSignals(BaseModel):
    """Structured evaluation output returned by the LLM."""

    fit_rationale: str
    must_have_coverage: float = Field(ge=0.0, le=1.0)
    nice_to_have_coverage: float = Field(ge=0.0, le=1.0)
    location_compatibility: float = Field(ge=0.0, le=1.0)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    company_context: dict[str, Any] = Field(default_factory=dict)


def compute_fit_score(
    *,
    must_have_coverage: float,
    nice_to_have_coverage: float,
    location_compatibility: float,
    blocking_gaps_count: int,
    has_surviving_claims: bool,
) -> float:
    """
    Pure deterministic function calculating fit score from structured sub-signals.

    Guarantees:
    - Same inputs => same score (pure function).
    - Must-have coverage: 50%, Nice-to-have: 20%, Location: 30%.
    - Penalty of 0.25 per blocking gap.
    - Capped at 0.30 if not backed by at least one real surviving evidence claim.
    - Strictly clamped to [0.0, 1.0].
    """
    base = 0.50 * must_have_coverage + 0.20 * nice_to_have_coverage + 0.30 * location_compatibility

    penalty = 0.25 * blocking_gaps_count
    score = max(0.0, min(1.0, base - penalty))

    if not has_surviving_claims:
        score = min(score, 0.30)

    return round(score, 4)


def derive_recommended_action(fit_score: float, has_blocking_gaps: bool) -> str:
    """
    Derive recommended action strictly by explicit business rules.

    Action in {apply_now, prepare_then_apply, research_first, skip}.
    A role with any blocking must-have gap never returns 'apply_now'.
    """
    if has_blocking_gaps:
        if fit_score >= 0.50:
            return "prepare_then_apply"
        return "skip"

    if fit_score >= 0.80:
        return "apply_now"
    elif fit_score >= 0.60:
        return "prepare_then_apply"
    elif fit_score >= 0.40:
        return "research_first"
    else:
        return "skip"


class RoleAssessor:
    """Evaluates job opening fit against career goals using grounded candidate evidence claims."""

    def __init__(
        self,
        llm: StructuredLLMClient,
        embedder: Any | None = None,
        assessor_version: str = "v1",
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.assessor_version = assessor_version

    def assess(
        self,
        *,
        role: Role,
        goal: Goal,
        claims: list[EvidenceClaim],
    ) -> RoleAssessmentCreate:
        """
        Assess a role against a target career goal with grounded candidate evidence claims.

        Discards fabricated claim IDs, audits blocking must-haves, computes an auditable fit score,
        and derives the recommended action.
        """
        # Map valid claim IDs for strict grounding enforcement
        valid_claims_map: dict[str, EvidenceClaim] = {str(c.id): c for c in claims}

        claims_listing = []
        for c in claims:
            claims_listing.append(
                f"- ID: {c.id}\n"
                f"  Claim: {c.claim}\n"
                f"  Kind: {c.source} | Excerpt: {c.source_excerpt}"
            )
        claims_text = "\n".join(claims_listing) if claims_listing else "(No evidence claims found)"

        target_spec = goal.target_spec or {}
        must_have = target_spec.get("must_have", [])
        nice_to_have = target_spec.get("nice_to_have", [])
        unstated_but_real = target_spec.get("unstated_but_real", [])

        system_prompt = (
            "You are an expert career intelligence assessor. Evaluate the suitability of a target job role "
            "against the candidate's career goal and verified evidence claims.\n\n"
            "STRICT GROUNDING CONSTRAINTS:\n"
            "1. You MUST only cite claim IDs from the provided candidate evidence list in 'supporting_claim_ids'. "
            "Never invent or hallucinate claim IDs.\n"
            "2. If a role requirement is a must-have but the candidate has ZERO supporting claims, you MUST mark "
            "that skill gap as blocking: true.\n"
            "3. In 'company_context', extract only factual details found in the job posting or verified company name/location. "
            "Do NOT speculate on headcount, funding, or culture.\n"
            "4. Return structured sub-signals: must_have_coverage (0.0 to 1.0), nice_to_have_coverage (0.0 to 1.0), "
            "and location_compatibility (0.0 to 1.0)."
        )

        user_prompt = (
            f"=== TARGET CAREER GOAL ===\n"
            f"Objective: {goal.objective_text}\n"
            f"Must-Have Requirements: {json.dumps(must_have)}\n"
            f"Nice-To-Have: {json.dumps(nice_to_have)}\n"
            f"Unstated But Real: {json.dumps(unstated_but_real)}\n\n"
            f"=== CANDIDATE EVIDENCE CLAIMS ===\n"
            f"{claims_text}\n\n"
            f"=== TARGET JOB ROLE ===\n"
            f"Role Title: {role.title}\n"
            f"Company: {role.company.name if role.company else 'Unknown'}\n"
            f"Location: {role.location} ({role.location_type})\n"
            f"Posting URL: {role.posting_url}\n"
            f"Requirements / Description:\n"
            f"{role.requirements_summary or 'No description provided'}\n\n"
            f"Evaluate the fit and return the structured sub-signals."
        )

        eval_result: LLMRoleSubSignals = self.llm.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=LLMRoleSubSignals,
        )

        # Grounding Audit: Discard any fabricated claim IDs
        surviving_claim_ids: list[UUID] = []
        for cid_str in eval_result.supporting_claim_ids:
            clean_str = cid_str.strip()
            if clean_str in valid_claims_map:
                try:
                    surviving_claim_ids.append(UUID(clean_str))
                except ValueError:
                    pass

        has_surviving = len(surviving_claim_ids) > 0

        # Check for blocking gaps
        blocking_gaps_count = sum(1 for g in eval_result.skill_gaps if g.blocking)
        has_blocking = blocking_gaps_count > 0

        # Deterministic Python score computation
        fit_score = compute_fit_score(
            must_have_coverage=eval_result.must_have_coverage,
            nice_to_have_coverage=eval_result.nice_to_have_coverage,
            location_compatibility=eval_result.location_compatibility,
            blocking_gaps_count=blocking_gaps_count,
            has_surviving_claims=has_surviving,
        )

        # Derive recommended action
        recommended_action = derive_recommended_action(
            fit_score=fit_score,
            has_blocking_gaps=has_blocking,
        )

        return RoleAssessmentCreate(
            role_id=role.id,
            goal_id=goal.id,
            fit_score=fit_score,
            fit_rationale=eval_result.fit_rationale,
            supporting_claim_ids=surviving_claim_ids,
            skill_gaps=eval_result.skill_gaps,
            company_context=eval_result.company_context,
            recommended_action=recommended_action,
            assessed_at=datetime.now(UTC),
            assessor_version=self.assessor_version,
        )
