"""Hypothesis tracking, evaluation, and empirical falsification."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Action, ActionOutcome, Cycle, Goal, StrategyNote, StrategyNoteStatus


def evaluate_hypotheses(
    session: Session,
    goal: Goal,
    *,
    evaluation_cycle_window: int = 3,
    now: datetime | None = None,
) -> list[StrategyNote]:
    """
    Evaluate open strategy hypotheses whose evaluation horizon has elapsed.

    Transitions:
    - Confirmed / Lift achieved -> StrategyNoteStatus.GRADUATED
    - Refuted / Failed threshold -> StrategyNoteStatus.REFUTED (feeds into detect pass)
    - Horizon not yet reached -> remains StrategyNoteStatus.ACTIVE
    """
    _ = now or datetime.now(UTC)

    # 1. Fetch active strategy notes for this goal
    stmt = select(StrategyNote).where(
        StrategyNote.goal_id == goal.id,
        StrategyNote.status == StrategyNoteStatus.ACTIVE,
    )
    active_notes = session.execute(stmt).scalars().all()
    if not active_notes:
        return []

    # 2. Get max cycle number for goal
    max_c = (
        session.scalar(
            select(Cycle.cycle_number)
            .where(Cycle.goal_id == goal.id)
            .order_by(Cycle.cycle_number.desc())
        )
        or 0
    )

    evaluated: list[StrategyNote] = []

    for note in active_notes:
        # Determine starting cycle for this note
        start_cycle_num = 1
        if note.cycle:
            start_cycle_num = note.cycle.cycle_number
        elif note.cycle_id:
            c_row = session.get(Cycle, note.cycle_id)
            if c_row:
                start_cycle_num = c_row.cycle_number

        cycles_elapsed = max_c - start_cycle_num
        if cycles_elapsed < evaluation_cycle_window:
            continue

        # Horizon elapsed -> evaluate performance under this strategy
        outcomes_stmt = (
            select(ActionOutcome.binary_success)
            .join(Action, Action.id == ActionOutcome.action_id)
            .where(Action.strategy_id == note.strategy_id)
        )
        outcomes = session.execute(outcomes_stmt).scalars().all()

        if len(outcomes) >= 3:
            success_rate = sum(1 for o in outcomes if o) / len(outcomes)
            # If conversion is below 5%, hypothesis was refuted
            if success_rate < 0.05:
                note.status = StrategyNoteStatus.REFUTED
                note.reflection = (
                    f"Hypothesis refuted after {cycles_elapsed} cycles: observed conversion rate "
                    f"was {success_rate:.1%} ({sum(1 for o in outcomes if o)}/{len(outcomes)})."
                )
            else:
                note.status = StrategyNoteStatus.GRADUATED
                note.reflection = (
                    f"Hypothesis confirmed/graduated after {cycles_elapsed} cycles with "
                    f"{success_rate:.1%} conversion rate."
                )
            evaluated.append(note)

    session.flush()
    return evaluated
