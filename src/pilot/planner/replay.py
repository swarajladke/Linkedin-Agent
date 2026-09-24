"""Counterfactual replay harness for Pilot Planner cycles."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Action, Cycle, Goal
from pilot.planner.diagnose import diagnose
from pilot.planner.generate import generate_actions
from pilot.planner.observe import observe
from pilot.planner.schemas import ReplayResult
from pilot.planner.score import score_actions, select_actions


def replay_cycle(
    session: Session,
    cycle_id: UUID,
    *,
    scorer=None,
    min_fit: float | None = None,
    max_actions: int | None = None,
) -> ReplayResult:
    """
    Re-evaluate a stored cycle observation under an altered policy WITHOUT writing any DB state.

    Useful for:
    - Counterfactual analysis: 'What would have been selected with min_fit=0.80 instead of 0.50?'
    - Scorer experimentation: swapping in a new scoring function without side effects.

    Returns:
    - ReplayResult with diff between original and counterfactual action sets.
    """
    # 1. Load the historical cycle and its goal
    cycle = session.get(Cycle, cycle_id)
    if not cycle:
        raise ValueError(f"Cycle {cycle_id} not found.")

    goal = session.get(Goal, cycle.goal_id)
    if not goal:
        raise ValueError(f"Goal {cycle.goal_id} not found.")

    # 2. Load original selected actions
    original_actions = (
        session.execute(select(Action).where(Action.cycle_id == cycle_id)).scalars().all()
    )
    original_ids = [a.id for a in original_actions]

    # 3. Re-observe from live DB at historical timestamp
    if cycle.started_at:
        try:
            replay_now = cycle.started_at
            fresh_obs = observe(session, goal, now=replay_now)
            observation = fresh_obs
        except Exception:
            # If observation fails, skip replay gracefully
            return ReplayResult(
                cycle_id=cycle_id,
                cycle_number=cycle.cycle_number,
                original_action_ids=original_ids,
                counterfactual_action_ids=[],
                added_action_ids=[],
                removed_action_ids=original_ids,
                rationale_diff="Replay failed: could not reconstruct observation.",
            )
    else:
        return ReplayResult(
            cycle_id=cycle_id,
            cycle_number=cycle.cycle_number,
            original_action_ids=original_ids,
            counterfactual_action_ids=[],
            added_action_ids=[],
            removed_action_ids=original_ids,
            rationale_diff="Replay failed: cycle has no started_at timestamp.",
        )

    # 4. Re-diagnose from stored cycle observation
    diagnosis = diagnose(goal, observation)

    # 5. Re-generate candidates (may differ if DB state changed, e.g. new assessments)
    candidates = generate_actions(session, goal, diagnosis)

    # Apply min_fit filter if specified
    if min_fit is not None:
        candidates = [c for c in candidates if c.fit_score >= min_fit]

    # 6. Re-score with alternate scorer or default
    strategy_notes = getattr(goal, "strategy_notes", [])
    if scorer is not None:
        scored = scorer(candidates, diagnosis, strategy_notes)
    else:
        scored = score_actions(candidates, diagnosis, strategy_notes=strategy_notes)

    # 7. Re-select with alternate budget
    selected = select_actions(scored, max_actions=max_actions)
    counterfactual_ids = [s.candidate.target_id for s in selected]

    # 8. Compute diff
    original_set = set(original_ids)
    counterfactual_set = set(counterfactual_ids)
    added = list(counterfactual_set - original_set)
    removed = list(original_set - counterfactual_set)

    rationale_parts = []
    if min_fit is not None:
        rationale_parts.append(f"Raised fit floor to {min_fit:.2f}")
    if max_actions is not None:
        rationale_parts.append(f"Changed max_actions to {max_actions}")
    if scorer is not None:
        rationale_parts.append("Applied custom scorer")
    if not rationale_parts:
        rationale_parts.append("No policy changes (identity replay)")

    rationale_diff = (
        f"Counterfactual replay of Cycle #{cycle.cycle_number}: "
        f"{', '.join(rationale_parts)}. "
        f"Original: {len(original_ids)} actions; Counterfactual: {len(counterfactual_ids)} actions. "
        f"Added {len(added)}, Removed {len(removed)}."
    )

    return ReplayResult(
        cycle_id=cycle_id,
        cycle_number=cycle.cycle_number,
        original_action_ids=original_ids,
        counterfactual_action_ids=counterfactual_ids,
        added_action_ids=added,
        removed_action_ids=removed,
        rationale_diff=rationale_diff,
    )
