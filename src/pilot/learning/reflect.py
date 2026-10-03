"""Constrained, bounded LLM strategy reflection proposing falsifiable policy adjustments."""

import json
from typing import Any

from pydantic import BaseModel, Field

from pilot.db.models import Goal, Strategy
from pilot.extraction.llm import StructuredLLMClient
from pilot.learning.schemas import FailureSignal, PolicyChange, Reflection

# Strict allow-list of tunable policy parameters and their mathematical delta bounds
POLICY_PARAMETER_BOUNDS: dict[str, dict[str, Any]] = {
    "fit_floor": {
        "min": 0.30,
        "max": 0.90,
        "max_delta": 0.15,
        "type": float,
        "default": 0.50,
    },
    "elevated_fit_floor": {
        "min": 0.50,
        "max": 0.95,
        "max_delta": 0.15,
        "type": float,
        "default": 0.75,
    },
    "conversion_prior_factor": {
        "min": 0.10,
        "max": 1.00,
        "max_delta": 0.20,
        "type": float,
        "default": 0.40,
    },
    "default_conversion_prior": {
        "min": 0.10,
        "max": 0.90,
        "max_delta": 0.20,
        "type": float,
        "default": 0.70,
    },
    "max_applications_per_day": {
        "min": 1,
        "max": 50,
        "max_delta": 5,
        "type": int,
        "default": 10,
    },
    "horizon_days": {
        "min": 3,
        "max": 60,
        "max_delta": 7,
        "type": int,
        "default": 14,
    },
}

MAX_POLICY_CHANGES_PER_REFLECTION = 2


class RawPolicyChangeProposal(BaseModel):
    parameter: str
    new_value: float | int


class RawReflectionProposal(BaseModel):
    explanation: str = Field(description="Causal reasoning linking the failure signal to policy")
    hypothesis: str = Field(
        description="Falsifiable hypothesis with concrete predicted lift and horizon"
    )
    proposed_changes: list[RawPolicyChangeProposal] = Field(default_factory=list)


def validate_and_build_policy_changes(
    current_policy: dict[str, Any],
    proposed: list[RawPolicyChangeProposal],
) -> list[PolicyChange]:
    """
    Validate proposed policy changes against the strict allow-list and bounds.

    Rules:
    - At most 2 parameters changed per reflection.
    - Parameter must be in allow-list.
    - Parameter delta must not exceed max_delta.
    - New value must be within [min, max].
    - Reject on any violation; NEVER clamp.
    """
    if len(proposed) > MAX_POLICY_CHANGES_PER_REFLECTION:
        raise ValueError(
            f"Reflection proposed {len(proposed)} changes, exceeding the limit of "
            f"{MAX_POLICY_CHANGES_PER_REFLECTION} parameters."
        )

    validated_changes: list[PolicyChange] = []

    for item in proposed:
        param = item.parameter
        if param not in POLICY_PARAMETER_BOUNDS:
            raise ValueError(
                f"Parameter '{param}' is outside the allowed tunable policy parameters."
            )

        spec = POLICY_PARAMETER_BOUNDS[param]
        expected_type = spec["type"]
        new_val = expected_type(item.new_value)
        old_val = expected_type(current_policy.get(param, spec["default"]))

        delta = round(new_val - old_val, 4)
        max_delta = spec["max_delta"]

        if abs(delta) > max_delta:
            raise ValueError(
                f"Proposed delta {delta:+g} for '{param}' exceeds maximum allowed delta ±{max_delta}."
            )

        if not (spec["min"] <= new_val <= spec["max"]):
            raise ValueError(
                f"Proposed new value {new_val} for '{param}' is outside bounds [{spec['min']}, {spec['max']}]."
            )

        validated_changes.append(
            PolicyChange(
                parameter=param,
                old_value=old_val,
                new_value=new_val,
                delta=delta,
            )
        )

    return validated_changes


def reflect(
    goal: Goal,
    strategy: Strategy,
    signal: FailureSignal,
    observation_history: list[dict[str, Any]],
    llm: StructuredLLMClient | None = None,
) -> Reflection:
    """
    Generate bounded policy reflection and hypothesis from failure signal using structured LLM.

    The LLM proposes policy adjustments, but output is strictly validated against
    the allow-list and delta limits. Unbounded or disallowed adjustments are rejected.
    """
    current_policy = strategy.policy or {}

    system_prompt = (
        "You are the strategic reflection engine for an autonomous career agent.\n"
        "You analyze deterministic failure signals and diagnose why the current strategy failed.\n"
        "Propose small, bounded adjustments to tunable policy parameters to overcome the failure.\n"
        "Rules:\n"
        f"- You may change at most {MAX_POLICY_CHANGES_PER_REFLECTION} parameters.\n"
        f"- Allowed parameters: {list(POLICY_PARAMETER_BOUNDS.keys())}.\n"
        "- Every change must be small and bounded.\n"
        "- Formulate a strictly falsifiable hypothesis sentence with concrete thresholds and cycle horizon.\n"
    )

    user_prompt = (
        f"Goal: {goal.objective_text}\n"
        f"Current Strategy Version: {strategy.version}\n"
        f"Current Policy:\n{json.dumps(current_policy, indent=2)}\n\n"
        f"Failure Signal:\n{json.dumps(signal.model_dump(mode='json'), indent=2)}\n\n"
        f"Recent Observations Summary:\n{json.dumps(observation_history[-3:], default=str, indent=2) if observation_history else 'None'}\n\n"
        "Propose a structured reflection."
    )

    if llm is None:
        # Deterministic heuristic proposal if LLM client is omitted
        param = (
            "fit_floor"
            if signal.kind.value in ("credible_interval_breached", "consecutive_starved")
            else "conversion_prior_factor"
        )
        old_val = float(current_policy.get(param, 0.50 if param == "fit_floor" else 0.40))
        delta = 0.10 if param == "fit_floor" else -0.10
        raw_proposal = RawReflectionProposal(
            explanation=f"Signal {signal.kind.value} indicates strategy sub-optimality. Adjusting {param}.",
            hypothesis=f"Adjusting {param} by {delta:+g} will improve conversion within 3 cycles.",
            proposed_changes=[
                RawPolicyChangeProposal(parameter=param, new_value=round(old_val + delta, 4))
            ],
        )
    else:
        raw_proposal = llm.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=RawReflectionProposal,
        )

    validated_changes = validate_and_build_policy_changes(
        current_policy=current_policy,
        proposed=raw_proposal.proposed_changes,
    )

    return Reflection(
        signal=signal,
        explanation=raw_proposal.explanation,
        hypothesis=raw_proposal.hypothesis,
        proposed_changes=validated_changes,
    )
