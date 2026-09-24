"""Goal compiler translating free-text objectives into structured goals with back-solved timelines."""

import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pilot.db.models import GoalStatus
from pilot.extraction.llm import StructuredLLMClient
from pilot.goals.errors import InfeasibleGoalError
from pilot.goals.schemas import CompiledGoalDraft
from pilot.schemas.goal import GoalCreate, SubGoal

GOAL_COMPILER_SYSTEM_PROMPT = """You are Pilot's Goal Compiler.
Your role is to translate a candidate's free-text career objective and constraints into a machine-checkable goal specification.

Decompose the objective into target_spec:
1. must_have: Explicit non-negotiable requirements mentioned in objective or constraints (e.g. role title, location, compensation floor).
2. nice_to_have: Stated preferences or flexibilities (e.g. preferred tech stack, company size, perks).
3. unstated_but_real: Implicit table-stakes and industry baseline prerequisites for this role that the user never wrote (e.g. 'can pass a DSA / system design screen', 'public code artifacts or verifiable GitHub history', 'working hours / timezone overlap').

Propose success_criteria:
- MUST BE NUMERIC ONLY (e.g. min_offers, min_salary, target_applications, min_response_rate).
- Do not include qualitative strings or booleans here.

Propose realistic funnel_assumptions (probabilities in (0, 1]):
- application_to_response: Expected conversion from application to recruiter/company response.
- response_to_interview: Expected conversion from response to first round interview.
- interview_to_offer: Expected conversion from interview stage to formal job offer.
"""


class GoalCompiler:
    """Compiles free-text career objectives and constraints into structured, verified Goal objects."""

    def __init__(self, llm: StructuredLLMClient, now: datetime | None = None) -> None:
        self.llm = llm
        self.now = now

    def compile(
        self,
        *,
        user_id: UUID,
        objective_text: str,
        constraints: dict[str, Any],
        deadline: datetime,
    ) -> GoalCreate:
        """Compile a goal from objective text and constraints with deterministic back-solved sub-goals.

        Raises InfeasibleGoalError if deadline is in the past or required volume exceeds rate constraints.
        """
        ref_now = self.now or datetime.now(UTC)

        # Align timezone awareness to avoid subtraction/comparison type errors
        if deadline.tzinfo is not None and ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=deadline.tzinfo)
        elif deadline.tzinfo is None and ref_now.tzinfo is not None:
            deadline = deadline.replace(tzinfo=ref_now.tzinfo)

        if deadline <= ref_now:
            raise InfeasibleGoalError(
                f"Goal deadline {deadline.isoformat()} is in the past relative to {ref_now.isoformat()}."
            )

        user_prompt = (
            f"Objective: {objective_text.strip()}\n"
            f"Constraints: {constraints}\n"
            f"Deadline: {deadline.isoformat()}\n"
            f"Current Time: {ref_now.isoformat()}"
        )

        draft = self.llm.complete_structured(
            system_prompt=GOAL_COMPILER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=CompiledGoalDraft,
        )

        # Post-validation: strip any non-numeric values from success_criteria
        clean_criteria: dict[str, int | float] = {}
        for k, v in draft.success_criteria.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, int | float):
                clean_criteria[k] = v
            elif isinstance(v, str):
                try:
                    parsed_val = float(v)
                    clean_criteria[k] = int(parsed_val) if parsed_val.is_integer() else parsed_val
                except (ValueError, TypeError):
                    continue

        min_offers = int(clean_criteria.get("min_offers", 1))
        if min_offers < 1:
            min_offers = 1
        clean_criteria["min_offers"] = min_offers

        # Deterministic Python back-solving of required funnel volumes
        p_io = draft.funnel_assumptions.interview_to_offer
        p_ri = draft.funnel_assumptions.response_to_interview
        p_ar = draft.funnel_assumptions.application_to_response
        p_sa = draft.funnel_assumptions.sourcing_to_application or 0.5

        interviews_needed = math.ceil(min_offers / p_io)
        responses_needed = math.ceil(interviews_needed / p_ri)
        applications_needed = math.ceil(responses_needed / p_ar)
        roles_to_source = math.ceil(applications_needed / p_sa)

        # Reflect computed volumes in success_criteria
        if "target_applications" in clean_criteria:
            clean_criteria["target_applications"] = max(
                clean_criteria["target_applications"], applications_needed
            )
        else:
            clean_criteria["target_applications"] = applications_needed

        # Feasibility check against rate caps in constraints
        available_days = (deadline - ref_now).total_seconds() / 86400.0
        max_apps_per_day = constraints.get("max_applications_per_day")
        if max_apps_per_day is None:
            max_apps_per_day = constraints.get("max_apps_per_day")

        if max_apps_per_day is not None:
            try:
                max_rate = float(max_apps_per_day)
                if max_rate <= 0 or (applications_needed > max_rate * available_days):
                    raise InfeasibleGoalError(
                        f"Back-solved volume requires {applications_needed} applications over "
                        f"{available_days:.1f} days ({applications_needed / max(available_days, 1e-6):.2f}/day), "
                        f"exceeding constraint of {max_rate}/day."
                    )
            except (ValueError, TypeError):
                pass

        # Sub-goal timeline phase allocation strictly between now and deadline
        total_window = deadline - ref_now

        sg_1 = SubGoal(
            id="sg_1",
            title="Evidence & Portfolio Readiness",
            metric_target={"evidence_claims_verified": 10},
            deadline=ref_now + total_window * 0.20,
            status="pending",
            depends_on=[],
        )
        sg_2 = SubGoal(
            id="sg_2",
            title="Role Discovery and Sourcing Pipeline",
            metric_target={"roles_identified": roles_to_source},
            deadline=ref_now + total_window * 0.45,
            status="pending",
            depends_on=["sg_1"],
        )
        sg_3 = SubGoal(
            id="sg_3",
            title="Outreach and Application Submissions",
            metric_target={"applications_submitted": applications_needed},
            deadline=ref_now + total_window * 0.70,
            status="pending",
            depends_on=["sg_2"],
        )
        sg_4 = SubGoal(
            id="sg_4",
            title="Interview Conversions and Offer Pursuit",
            metric_target={
                "interviews_completed": interviews_needed,
                "offers_received": min_offers,
            },
            deadline=ref_now + total_window * 0.90,
            status="pending",
            depends_on=["sg_3"],
        )

        return GoalCreate(
            user_id=user_id,
            objective_text=objective_text,
            constraints_json=constraints,
            target_spec=draft.target_spec,
            success_criteria=clean_criteria,
            sub_goals=[sg_1, sg_2, sg_3, sg_4],
            deadline=deadline,
            status=GoalStatus.ACTIVE,
        )
