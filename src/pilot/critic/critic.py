"""Pilot Critic verification gate coordinating grounding, voice, and factual checks."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from pilot.critic.checks import factual_check, grounding_check, voice_check
from pilot.critic.schemas import (
    CriticFailure,
    CriticReviewRecord,
    CriticVerdict,
    FailureSeverity,
    VoiceProfile,
)
from pilot.db.models import EvidenceClaim, Role
from pilot.extraction.llm import StructuredLLMClient


class Critic:
    """Mandatory verification gate sitting between artifact drafting and shipment."""

    def __init__(self, llm: StructuredLLMClient | None = None) -> None:
        self.llm = llm

    def _extract_text(self, artifact: str | dict[str, Any] | BaseModel) -> str:
        """Extract clean text representation from draft application package or string."""
        if isinstance(artifact, str):
            # Check if it's JSON
            try:
                data = json.loads(artifact)
                if isinstance(data, dict):
                    return self._extract_dict_text(data)
            except Exception:
                pass
            return artifact
        elif isinstance(artifact, BaseModel):
            return self._extract_dict_text(artifact.model_dump())
        elif isinstance(artifact, dict):
            return self._extract_dict_text(artifact)
        return str(artifact)

    def _extract_dict_text(self, data: dict[str, Any]) -> str:
        parts: list[str] = []
        if "tailored_summary" in data:
            parts.append(str(data["tailored_summary"]))
        if "tailored_bullets" in data and isinstance(data["tailored_bullets"], list):
            parts.extend(str(b) for b in data["tailored_bullets"])
        if not parts:
            for v in data.values():
                if isinstance(v, str):
                    parts.append(v)
        return "\n".join(parts)

    def review(
        self,
        *,
        artifact: str | dict[str, Any] | BaseModel,
        claims: Sequence[EvidenceClaim],
        profile: VoiceProfile,
        role: Role | None = None,
        attempt: int = 1,
        action_id: UUID | None = None,
    ) -> CriticReviewRecord:
        """
        Evaluate draft artifact through grounding, voice, and factual checks.

        Guarantees:
        - Runs all three checks always, even after the first blocking failure.
        - any blocking failure and attempt < 2 -> regenerate
        - any blocking failure and attempt >= 2 -> drop
        - only advisory failures -> pass, with failures recorded
        - clean -> pass
        """
        artifact_text = self._extract_text(artifact)

        # Execute all three checks unconditionally
        g_res = grounding_check(artifact_text, claims, llm=self.llm)
        v_res = voice_check(artifact_text, profile)
        f_res = factual_check(artifact_text, claims, role)

        all_failures: list[CriticFailure] = g_res.failures + v_res.failures + f_res.failures
        has_blocking = any(f.severity == FailureSeverity.BLOCKING for f in all_failures)

        if has_blocking:
            if attempt < 2:
                verdict = CriticVerdict.REGENERATE
            else:
                verdict = CriticVerdict.DROP
        else:
            verdict = CriticVerdict.PASS

        # Generate a placeholder UUID if action_id is None
        review_action_id = action_id or UUID("00000000-0000-0000-0000-000000000000")

        return CriticReviewRecord(
            action_id=review_action_id,
            attempt=attempt,
            artifact_text=artifact_text,
            grounding_passed=g_res.passed,
            voice_passed=v_res.passed,
            factual_passed=f_res.passed,
            verdict=verdict,
            failures=all_failures,
            reviewed_at=datetime.now(UTC),
        )
