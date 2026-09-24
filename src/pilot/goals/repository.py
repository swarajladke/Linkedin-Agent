"""Database persistence for compiled goals, initial strategy v1, and baseline hypothesis note."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Goal, GoalStatus, Strategy, StrategyNote, StrategyNoteStatus
from pilot.schemas.goal import GoalCreate


def persist_compiled_goal(session: Session, goal: GoalCreate) -> tuple[Goal, Strategy]:
    """Persist a compiled goal, create initial strategy version 1, and write baseline hypothesis.

    Enforces a single ACTIVE goal per user: any existing ACTIVE goals for the user
    are transitioned to GoalStatus.PAUSED before inserting the new goal.
    """
    # Enforce single ACTIVE goal per user: pause prior active goals
    existing_active = session.scalars(
        select(Goal).where(
            Goal.user_id == goal.user_id,
            Goal.status == GoalStatus.ACTIVE,
        )
    ).all()
    for prev_goal in existing_active:
        prev_goal.status = GoalStatus.PAUSED

    # Insert Goal row
    db_goal = Goal(
        user_id=goal.user_id,
        objective_text=goal.objective_text,
        constraints_json=goal.constraints_json,
        target_spec=goal.target_spec.model_dump(),
        success_criteria=goal.success_criteria,
        sub_goals=[sg.model_dump(mode="json") for sg in goal.sub_goals],
        deadline=goal.deadline,
        status=goal.status,
    )
    session.add(db_goal)
    session.flush()

    # Create Strategy version 1
    db_strategy = Strategy(
        goal_id=db_goal.id,
        version=1,
        parent_version_id=None,
    )
    session.add(db_strategy)
    session.flush()

    # Insert baseline StrategyNote capturing initial funnel assumptions and back-solved volumes
    sub_goal_summary = ", ".join(
        f"{sg.id}: {sg.title} (targets: {sg.metric_target})" for sg in goal.sub_goals
    )
    hypothesis = (
        f"Baseline Strategy v1 for objective: '{goal.objective_text}'. "
        f"Success criteria: {goal.success_criteria}. "
        f"Back-solved milestone plan: [{sub_goal_summary}]."
    )
    db_note = StrategyNote(
        goal_id=db_goal.id,
        strategy_id=db_strategy.id,
        hypothesis=hypothesis,
        status=StrategyNoteStatus.ACTIVE,
    )
    session.add(db_note)
    session.flush()

    return db_goal, db_strategy
