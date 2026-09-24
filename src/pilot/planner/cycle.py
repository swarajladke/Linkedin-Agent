"""Cycle orchestration for Pilot Planner."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    Application,
    ApplicationStage,
    Cycle,
    Goal,
    Strategy,
)
from pilot.extraction.llm import StructuredLLMClient
from pilot.planner.diagnose import diagnose
from pilot.planner.execute import execute
from pilot.planner.generate import generate_actions
from pilot.planner.observe import observe
from pilot.planner.predict import record_prediction
from pilot.planner.schemas import CycleResult, ExecutionResult
from pilot.planner.score import score_actions, select_actions


def run_cycle(
    session: Session,
    goal: Goal,
    *,
    now: datetime,
    dry_run: bool = False,
    llm: StructuredLLMClient | None = None,
) -> CycleResult:
    """
    Orchestrate a single decision cycle:
    observe -> diagnose -> generate -> score -> select -> predict -> execute -> commit.

    Guarantees:
    - Atomicity: one transaction. Failure rolls back all writes; zero orphan actions.
    - Dry run: evaluates the full decision pipeline without persisting any DB state.
    - Persisted Prediction: prediction is recorded before action execution begins.
    - Invariant: action.created_at <= action.executed_at.
    """
    # 1. Determine next cycle number
    max_cycle_stmt = select(func.max(Cycle.cycle_number)).where(Cycle.goal_id == goal.id)
    current_max = session.scalar(max_cycle_stmt) or 0
    cycle_number = current_max + 1

    # 2. Observe & Diagnose
    observation = observe(session, goal, now=now)
    diagnosis = diagnose(goal, observation, llm=llm)

    # 3. Generate Actions
    candidates = generate_actions(session, goal, diagnosis)

    # 4. Score Actions
    strategy_notes = getattr(goal, "strategy_notes", [])
    scored = score_actions(candidates, diagnosis, strategy_notes=strategy_notes)

    # 5. Select Actions based on Constraints
    constraints = goal.constraints_json or {}
    max_apps_val = constraints.get("max_applications_per_day", constraints.get("max_apps_per_day"))
    max_apps: int | None = None
    if max_apps_val is not None:
        try:
            max_apps = int(max_apps_val)
        except (ValueError, TypeError):
            pass

    selected = select_actions(scored, max_actions=max_apps)

    if dry_run:
        return CycleResult(
            cycle_id=None,
            cycle_number=cycle_number,
            started_at=now,
            completed_at=now,
            dry_run=True,
            observation=observation,
            diagnosis=diagnosis,
            actions_proposed=len(candidates),
            actions_selected=len(selected),
            selected_actions=selected,
            execution_results=[],
        )

    # 6. Atomic Transaction Execution
    try:
        # Resolve active strategy version for goal
        strat_stmt = (
            select(Strategy)
            .where(Strategy.goal_id == goal.id, Strategy.retired_at.is_(None))
            .order_by(Strategy.version.desc())
        )
        active_strategy = session.execute(strat_stmt).scalars().first()
        strategy_id: UUID = (
            active_strategy.id
            if active_strategy
            else (goal.strategies[0].id if goal.strategies else None)
        )
        if not strategy_id:
            raise ValueError(f"Goal {goal.id} has no associated strategy.")

        # Create cycle row
        cycle = Cycle(
            goal_id=goal.id,
            cycle_number=cycle_number,
            strategy_id=strategy_id,
            started_at=now,
            observation=observation.model_dump(mode="json"),
            diagnosis=diagnosis.model_dump(mode="json"),
            actions_proposed=len(candidates),
            actions_selected=len(selected),
        )
        session.add(cycle)
        session.flush()

        execution_results: list[ExecutionResult] = []

        for item in selected:
            c = item.candidate

            # Build action row
            action = Action(
                goal_id=goal.id,
                strategy_id=strategy_id,
                cycle_id=cycle.id,
                action_type=c.action_type,
                target_type=c.target_type,
                target_id=c.target_id,
                reason=c.reason,
                predicted_outcome=item.predicted_outcome,
                predicted_probability=item.predicted_probability,
                cost=item.cost,
                created_at=now,
            )

            # Persist prediction structurally BEFORE execution
            action = record_prediction(
                session,
                action,
                cycle_id=cycle.id,
                prior=item.predicted_probability,
            )

            # Execute action
            exec_res = execute(session, action, now=now)
            execution_results.append(exec_res)

            # Record candidate application in world model
            app = Application(
                goal_id=goal.id,
                role_id=c.role_id,
                stage=ApplicationStage.APPLIED,
                applied_at=now,
                notes=f"Applied via Pilot Cycle #{cycle_number}",
            )
            session.add(app)

        cycle.completed_at = now
        session.commit()

        return CycleResult(
            cycle_id=cycle.id,
            cycle_number=cycle_number,
            started_at=now,
            completed_at=now,
            dry_run=False,
            observation=observation,
            diagnosis=diagnosis,
            actions_proposed=len(candidates),
            actions_selected=len(selected),
            selected_actions=selected,
            execution_results=execution_results,
        )

    except Exception:
        session.rollback()
        raise
