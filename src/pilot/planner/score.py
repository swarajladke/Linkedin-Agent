"""Deterministic pure function action scoring and budget-constrained selection."""

from decimal import Decimal

from pilot.db.models import StrategyNote, StrategyNoteStatus
from pilot.planner.schemas import CandidateAction, Diagnosis, DiagnosisCategory, ScoredAction

# Default prior conversion factor from fit_score to interview probability
DEFAULT_CONVERSION_PRIOR_FACTOR = 0.40
DEFAULT_ACTION_COST = Decimal("0.0")


def score_actions(
    candidates: list[CandidateAction],
    diagnosis: Diagnosis,
    strategy_notes: list[StrategyNote] | None = None,
) -> list[ScoredAction]:
    """
    Score candidate actions via a deterministic pure expected-value function. No LLM calls.

    Score components:
    - Base utility: candidate fit_score
    - Stage urgency: amplified when funnel is starved
    - Strategy notes: adjustment bonus from active tactical notes
    - Cost: deducted from score

    Guarantees:
    - Strict determinism: shuffled inputs produce identical output scores and ordering.
    - Tie-breaking: deterministic lexicographical order on role_id.
    """
    # 1. Compute urgency multiplier from diagnosis
    urgency = 1.0
    if diagnosis.category == DiagnosisCategory.STARVED:
        if diagnosis.starved_stage == "applications":
            urgency = 1.25
        elif diagnosis.starved_stage == "responses":
            urgency = 1.10
        else:
            urgency = 1.05

    # 2. Strategy note adjustments
    strategy_bonus = 0.0
    if strategy_notes:
        active_notes = [
            n for n in strategy_notes if getattr(n, "status", None) == StrategyNoteStatus.ACTIVE
        ]
        # Active notes add a subtle tactical bonus (up to +0.05)
        strategy_bonus = min(0.05, 0.01 * len(active_notes))

    scored: list[ScoredAction] = []
    for c in candidates:
        cost = DEFAULT_ACTION_COST
        # Probability prior: derived monotonically from fit_score
        predicted_prob = round(
            min(0.95, max(0.05, c.fit_score * DEFAULT_CONVERSION_PRIOR_FACTOR)), 4
        )

        # Expected value score
        ev_score = round((c.fit_score * urgency) + strategy_bonus - float(cost), 4)

        # Falsifiable prediction sentence
        predicted_outcome = (
            f"Tailored application package for '{c.role_title}' at {c.company_name} "
            f"will advance past initial screening to interview within 14 days."
        )

        scored.append(
            ScoredAction(
                candidate=c,
                score=ev_score,
                urgency=urgency,
                cost=cost,
                predicted_probability=predicted_prob,
                predicted_outcome=predicted_outcome,
            )
        )

    # Sort deterministically: score descending, then role_id ascending
    scored.sort(key=lambda s: (-s.score, str(s.candidate.role_id)))
    return scored


def select_actions(
    scored: list[ScoredAction],
    budget: Decimal | float | None = None,
    max_actions: int | None = None,
) -> list[ScoredAction]:
    """
    Select actions respecting financial budget and per-cycle volume caps.

    - Greedily admits highest-scored actions first.
    - Halts immediately if total cost would exceed budget.
    - Halts if max_actions (e.g. max_applications_per_day) is reached.
    """
    max_cap = max_actions if max_actions is not None and max_actions > 0 else len(scored)
    budget_limit = Decimal(str(budget)) if budget is not None else None

    selected: list[ScoredAction] = []
    cumulative_cost = Decimal("0.0")

    for action in scored:
        if len(selected) >= max_cap:
            break

        if budget_limit is not None and (cumulative_cost + action.cost > budget_limit):
            continue

        selected.append(action)
        cumulative_cost += action.cost

    return selected
