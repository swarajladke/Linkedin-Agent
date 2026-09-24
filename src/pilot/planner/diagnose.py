"""Deterministic funnel pace diagnosis with structured LLM root-cause investigation."""

import math

from pilot.db.models import Goal, GoalStatus
from pilot.extraction.llm import StructuredLLMClient
from pilot.planner.schemas import (
    Diagnosis,
    DiagnosisCategory,
    Observation,
    RootCauseHypothesis,
    RootCauseKind,
)

# Minimum sample floor required per stage before concluding conversion starvation
MIN_SAMPLE_FLOOR = 5

DIAGNOSIS_SYSTEM_PROMPT = """You are the Senior Talent Diagnostic Engine for Pilot.
A conversion funnel stage for a candidate's career search has been identified as STARVED.
Your task is to analyze the observed metrics and propose 1 to 3 root-cause hypotheses explaining why this stage is starved.

STRICT RULES:
1. Every hypothesis MUST choose a cause from the allowed enum:
   - low_fit_targeting: Job targeting is too broad or applied roles have low fit.
   - weak_evidence_coverage: Applications lack sufficient grounded evidence/skills for target roles.
   - insufficient_volume: Top-of-funnel or application rate is too low to produce conversions.
   - constraint_too_narrow: Search constraints or location/compensation filters overly restrict matching roles.
   - timing: In market for too short a time, or typical recruiter review SLA has not elapsed.
2. Every hypothesis MUST cite an exact numeric metric from the provided observation (supporting_metric_name and supporting_metric_value).
3. Any hypothesis without an exact supporting number from the observation will be discarded by the validator.
"""


def _get_obs_metrics(observation: Observation) -> dict[str, float]:
    """Flatten all numeric metrics in the observation for validation."""
    metrics: dict[str, float] = {
        "roles_sourced": float(observation.funnel_counts.roles_sourced),
        "roles_assessed": float(observation.funnel_counts.roles_assessed),
        "applications": float(observation.funnel_counts.applications),
        "responses": float(observation.funnel_counts.responses),
        "interviews": float(observation.funnel_counts.interviews),
        "offers": float(observation.funnel_counts.offers),
        "sourcing_to_assessed": float(observation.conversion_rates.sourcing_to_assessed),
        "assessed_to_applied": float(observation.conversion_rates.assessed_to_applied),
        "applied_to_response": float(observation.conversion_rates.applied_to_response),
        "response_to_interview": float(observation.conversion_rates.response_to_interview),
        "interview_to_offer": float(observation.conversion_rates.interview_to_offer),
        "elapsed_days": float(observation.timeline.elapsed_days),
        "remaining_days": float(observation.timeline.remaining_days),
        "total_days": float(observation.timeline.total_days),
        "pace_fraction": float(observation.timeline.pace_fraction),
        "evidence_claims_count": float(observation.evidence_claims_count),
    }
    if observation.average_fit_score is not None:
        metrics["average_fit_score"] = float(observation.average_fit_score)
    return metrics


def _validate_hypothesis(hypothesis: RootCauseHypothesis, obs_metrics: dict[str, float]) -> bool:
    """Validate that the hypothesis is backed by an actual number in the observation."""
    name = hypothesis.supporting_metric_name.strip().lower()
    val = hypothesis.supporting_metric_value

    if not isinstance(val, int | float) or math.isnan(val) or math.isinf(val):
        return False

    # Check if metric name matches and value is close (within 0.01 tolerance)
    if name in obs_metrics:
        if abs(obs_metrics[name] - val) <= 0.02:
            return True

    # Alternatively check if value matches any metric in observation
    for _, actual_val in obs_metrics.items():
        if abs(actual_val - val) <= 0.001:
            return True

    return False


def _extract_stage_targets(goal: Goal) -> dict[str, float]:
    """Extract back-solved milestone targets for each funnel stage."""
    targets = {
        "sourcing": 64.0,
        "applications": 32.0,
        "responses": 8.0,
        "interviews": 4.0,
        "offers": 1.0,
    }

    # Extract from goal success_criteria
    if goal.success_criteria:
        if "min_offers" in goal.success_criteria:
            targets["offers"] = float(goal.success_criteria["min_offers"])
        if "target_applications" in goal.success_criteria:
            targets["applications"] = float(goal.success_criteria["target_applications"])

    # Extract from sub_goals
    sub_goals = goal.sub_goals or []
    for sg in sub_goals:
        metric_target = (
            sg.get("metric_target", {})
            if isinstance(sg, dict)
            else getattr(sg, "metric_target", {})
        )
        if "roles_identified" in metric_target:
            targets["sourcing"] = float(metric_target["roles_identified"])
        if "applications_submitted" in metric_target:
            targets["applications"] = float(metric_target["applications_submitted"])
        if "interviews_completed" in metric_target:
            targets["interviews"] = float(metric_target["interviews_completed"])
        if "offers_received" in metric_target:
            targets["offers"] = float(metric_target["offers_received"])

    # Implicit responses target if not specified
    if targets["interviews"] > 0 and targets["responses"] <= targets["interviews"]:
        targets["responses"] = targets["interviews"] * 2.0

    return targets


def diagnose(
    goal: Goal,
    observation: Observation,
    *,
    llm: StructuredLLMClient | None = None,
    min_sample_floor: int = MIN_SAMPLE_FLOOR,
) -> Diagnosis:
    """
    Diagnose progress toward a career goal.

    Returns exactly one of:
    - on_pace
    - starved(stage)
    - blocked(reason)
    - constraint_conflict

    Stage classification is 100% deterministic Python walking the funnel bottom-up.
    LLM is used strictly for root-cause hypothesis generation on a starved stage.
    """
    # 1. BLOCKED Branch
    if goal.status == GoalStatus.PAUSED:
        return Diagnosis(
            category=DiagnosisCategory.BLOCKED,
            blocked_reason="Goal is currently paused",
        )
    if goal.status == GoalStatus.ABANDONED:
        return Diagnosis(
            category=DiagnosisCategory.BLOCKED,
            blocked_reason="Goal has been abandoned",
        )
    if goal.status == GoalStatus.ACHIEVED:
        return Diagnosis(
            category=DiagnosisCategory.BLOCKED,
            blocked_reason="Goal success criteria already achieved",
        )
    if observation.active_escalations_count > 0:
        return Diagnosis(
            category=DiagnosisCategory.BLOCKED,
            blocked_reason=f"{observation.active_escalations_count} unresolved escalation(s) require review",
        )
    if observation.evidence_claims_count == 0:
        return Diagnosis(
            category=DiagnosisCategory.BLOCKED,
            blocked_reason="No verified evidence claims available for application drafting",
        )

    # 2. CONSTRAINT_CONFLICT Branch
    constraints = goal.constraints_json or {}
    max_apps_val = constraints.get("max_applications_per_day", constraints.get("max_apps_per_day"))
    targets = _extract_stage_targets(goal)

    if max_apps_val is not None:
        try:
            max_apps = float(max_apps_val)
            if max_apps <= 0:
                return Diagnosis(
                    category=DiagnosisCategory.CONSTRAINT_CONFLICT,
                    conflict_details="Constraint max_applications_per_day is <= 0",
                )
            if observation.timeline.remaining_days > 0:
                target_apps = targets["applications"]
                remaining_apps = max(
                    0.0, target_apps - float(observation.funnel_counts.applications)
                )
                required_rate = remaining_apps / observation.timeline.remaining_days
                if required_rate > max_apps:
                    return Diagnosis(
                        category=DiagnosisCategory.CONSTRAINT_CONFLICT,
                        conflict_details=(
                            f"Required application rate ({required_rate:.1f}/day) exceeds "
                            f"max_applications_per_day constraint ({max_apps:.1f}/day)"
                        ),
                    )
        except (ValueError, TypeError):
            pass

    # 3. Deterministic Funnel Pace Evaluation (Bottom-Up)
    # Funnel stages from bottom to top:
    # offers -> interviews -> responses -> applications -> sourcing
    counts = observation.funnel_counts
    pace = observation.timeline.pace_fraction

    # If day 0 or very early, check if we have enough sample anywhere
    total_funnel_activity = (
        counts.roles_sourced
        + counts.applications
        + counts.responses
        + counts.interviews
        + counts.offers
    )

    stage_health: dict[str, str] = {}
    stage_evals = [
        {
            "stage": "offers",
            "actual": counts.offers,
            "target": targets["offers"],
            "input_sample": counts.interviews,
        },
        {
            "stage": "interviews",
            "actual": counts.interviews,
            "target": targets["interviews"],
            "input_sample": counts.responses,
        },
        {
            "stage": "responses",
            "actual": counts.responses,
            "target": targets["responses"],
            "input_sample": counts.applications,
        },
        {
            "stage": "applications",
            "actual": counts.applications,
            "target": targets["applications"],
            "input_sample": counts.roles_assessed
            if counts.roles_assessed > 0
            else counts.roles_sourced,
        },
        {
            "stage": "sourcing",
            "actual": counts.roles_sourced,
            "target": targets["sourcing"],
            "input_sample": counts.roles_sourced,
        },
    ]

    # Evaluate each stage status
    for entry in stage_evals:
        stage_name = entry["stage"]
        actual = entry["actual"]
        target = entry["target"]
        input_sample = entry["input_sample"]
        prorated_expected = max(1.0, math.ceil(target * pace)) if pace > 0 else 1.0

        if stage_name == "sourcing":
            # Sourcing is healthy if actual >= prorated_expected or actual >= min_sample_floor
            if actual >= prorated_expected or actual >= min_sample_floor:
                stage_health[stage_name] = "HEALTHY"
            elif observation.timeline.elapsed_days >= 3.0:
                stage_health[stage_name] = "STARVED"
            else:
                stage_health[stage_name] = "INSUFFICIENT_DATA"
        else:
            if input_sample < min_sample_floor:
                stage_health[stage_name] = "INSUFFICIENT_DATA"
            elif actual >= prorated_expected:
                stage_health[stage_name] = "HEALTHY"
            else:
                stage_health[stage_name] = "STARVED"

    # Walk bottom-up and find the FIRST stage that is STARVED
    starved_stage: str | None = None
    for entry in stage_evals:
        stage_name = entry["stage"]
        if stage_health.get(stage_name) == "STARVED":
            starved_stage = stage_name
            break

    # If no stage is starved:
    if not starved_stage:
        # Check if insufficient data across the board
        all_insufficient = total_funnel_activity < min_sample_floor or all(
            h == "INSUFFICIENT_DATA" for h in stage_health.values()
        )
        return Diagnosis(
            category=DiagnosisCategory.ON_PACE,
            stage_health=stage_health,
            insufficient_data=all_insufficient,
            details={"pace_fraction": pace, "targets": targets},
        )

    # 4. STARVED Branch: Root-Cause Investigation (LLM with deterministic fallback)
    obs_metrics = _get_obs_metrics(observation)
    hypotheses: list[RootCauseHypothesis] = []

    if llm is not None:
        user_prompt = (
            f"Goal: {goal.objective_text}\n"
            f"Starved Stage: {starved_stage}\n"
            f"Stage Health: {stage_health}\n"
            f"Observation Metrics: {obs_metrics}\n"
        )
        try:
            from pydantic import BaseModel

            class _LLMResponse(BaseModel):
                hypotheses: list[RootCauseHypothesis]

            draft_response = llm.complete_structured(
                system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=_LLMResponse,
            )
            # Filter hypotheses: discard any without exact numeric support
            for h in draft_response.hypotheses:
                if _validate_hypothesis(h, obs_metrics):
                    hypotheses.append(h)
        except Exception:
            # On LLM failure or parse error, fall back gracefully
            pass

    # If no hypotheses survived validation or no LLM was provided, generate deterministic fallback
    if not hypotheses:
        if starved_stage == "responses":
            hypotheses.append(
                RootCauseHypothesis(
                    cause=RootCauseKind.LOW_FIT_TARGETING,
                    explanation="Zero or below-pace responses despite sufficient application submissions.",
                    supporting_metric_name="applied_to_response",
                    supporting_metric_value=obs_metrics.get("applied_to_response", 0.0),
                )
            )
        elif starved_stage == "applications":
            hypotheses.append(
                RootCauseHypothesis(
                    cause=RootCauseKind.INSUFFICIENT_VOLUME,
                    explanation="Application submission volume is behind prorated target schedule.",
                    supporting_metric_name="applications",
                    supporting_metric_value=obs_metrics.get("applications", 0.0),
                )
            )
        elif starved_stage == "sourcing":
            hypotheses.append(
                RootCauseHypothesis(
                    cause=RootCauseKind.CONSTRAINT_TOO_NARROW,
                    explanation="Role sourcing volume is starved relative to funnel requirements.",
                    supporting_metric_name="roles_sourced",
                    supporting_metric_value=obs_metrics.get("roles_sourced", 0.0),
                )
            )
        else:
            hypotheses.append(
                RootCauseHypothesis(
                    cause=RootCauseKind.TIMING,
                    explanation=f"Stage {starved_stage} conversion is behind expected pace.",
                    supporting_metric_name="pace_fraction",
                    supporting_metric_value=obs_metrics.get("pace_fraction", 0.0),
                )
            )

    return Diagnosis(
        category=DiagnosisCategory.STARVED,
        starved_stage=starved_stage,
        stage_health=stage_health,
        hypotheses=hypotheses,
        insufficient_data=False,
        details={"pace_fraction": pace, "targets": targets},
    )
