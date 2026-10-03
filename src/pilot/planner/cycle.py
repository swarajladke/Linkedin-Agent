"""Cycle orchestration for Pilot Planner with closed-loop learning and calibration."""

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
from pilot.learning.adopt import evaluate_and_adopt
from pilot.learning.calibration import compute_calibration
from pilot.learning.detect import detect_strategy_failure
from pilot.learning.hypotheses import evaluate_hypotheses
from pilot.learning.outcomes import resolve_outcomes
from pilot.learning.reflect import reflect
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
    resolve_outcomes -> update calibration -> evaluate hypotheses -> observe -> diagnose
    -> generate -> score -> select -> predict (calibrated) -> execute (critic gate)
    -> detect -> (reflect -> evaluate_and_adopt) -> commit.

    Guarantees:
    - Atomicity: one transaction. Failure rolls back all writes; zero orphan actions.
    - Dry run: evaluates the full decision and learning pipeline without persisting any DB state.
    - Persisted Prediction: prediction is recorded before action execution begins.
    - Invariant: action.created_at <= action.executed_at.
    """
    # 1. Determine next cycle number
    max_cycle_stmt = select(func.max(Cycle.cycle_number)).where(Cycle.goal_id == goal.id)
    current_max = session.scalar(max_cycle_stmt) or 0
    cycle_number = current_max + 1

    # Resolve active strategy version for goal
    strat_stmt = (
        select(Strategy)
        .where(Strategy.goal_id == goal.id, Strategy.retired_at.is_(None))
        .order_by(Strategy.version.desc())
    )
    active_strategy = session.execute(strat_stmt).scalars().first()
    if not active_strategy and goal.strategies:
        active_strategy = goal.strategies[0]
    if not active_strategy:
        raise ValueError(f"Goal {goal.id} has no associated strategy.")

    strategy_id: UUID = active_strategy.id
    policy = active_strategy.policy or {}

    if dry_run:
        # Use savepoint so we can evaluate full learning loop without writing state
        savepoint = session.begin_nested()
        try:
            # 1. resolve outcomes
            res_result = resolve_outcomes(session, goal, now=now)
            # 2. update calibration
            cal_report = compute_calibration(session, strategy_id, now=now)
            # 3. evaluate hypotheses
            _ = evaluate_hypotheses(session, goal, now=now)
            # 4. observe & diagnose
            observation = observe(session, goal, now=now)
            diagnosis = diagnose(goal, observation, llm=llm)
            # 5. generate, score, select
            candidates = generate_actions(session, goal, diagnosis, policy=policy)
            strategy_notes = getattr(goal, "strategy_notes", [])
            scored = score_actions(
                candidates, diagnosis, strategy_notes=strategy_notes, policy=policy
            )
            constraints = goal.constraints_json or {}
            max_apps_val = constraints.get(
                "max_applications_per_day", constraints.get("max_apps_per_day")
            )
            max_apps: int | None = None
            if max_apps_val is not None:
                try:
                    max_apps = int(max_apps_val)
                except (ValueError, TypeError):
                    pass
            selected = select_actions(scored, max_actions=max_apps, policy=policy)

            # 6. detect failure & propose reflection/adoption
            failure_sig = detect_strategy_failure(session, goal, active_strategy, now=now)
            adopt_res = None
            if failure_sig:
                refl = reflect(
                    goal,
                    active_strategy,
                    failure_sig,
                    [observation.model_dump(mode="json")],
                    llm=llm,
                )
                adopt_res = evaluate_and_adopt(session, goal, refl, now=now)

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
                resolution_result=res_result,
                calibration_report=cal_report,
                failure_signal=failure_sig,
                adoption_result=adopt_res,
            )
        finally:
            savepoint.rollback()

    # Full Atomic Live Execution
    try:
        # Step 1: Resolve pending outcomes
        res_result = resolve_outcomes(session, goal, now=now)

        # Step 2: Update calibration
        cal_report = compute_calibration(session, strategy_id, now=now)

        # Step 3: Evaluate open hypotheses
        _ = evaluate_hypotheses(session, goal, now=now)

        # Step 4: Observe & Diagnose
        observation = observe(session, goal, now=now)
        diagnosis = diagnose(goal, observation, llm=llm)

        # Step 5: Generate actions
        candidates = generate_actions(session, goal, diagnosis, policy=policy)

        # Step 6: Score actions
        strategy_notes = getattr(goal, "strategy_notes", [])
        scored = score_actions(candidates, diagnosis, strategy_notes=strategy_notes, policy=policy)

        # Step 7: Select actions based on constraints & policy
        constraints = goal.constraints_json or {}
        max_apps_val = constraints.get(
            "max_applications_per_day", constraints.get("max_apps_per_day")
        )
        max_apps: int | None = None
        if max_apps_val is not None:
            try:
                max_apps = int(max_apps_val)
            except (ValueError, TypeError):
                pass

        selected = select_actions(scored, max_actions=max_apps, policy=policy)

        # Step 8: Create cycle row
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

        # Step 9: Predict (calibrated) & Execute through Critic Gate
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

            # Persist prediction structurally BEFORE execution through calibrated seam
            action = record_prediction(
                session,
                action,
                cycle_id=cycle.id,
                prior=item.predicted_probability,
            )

            # Execute action through Critic gate
            exec_res = execute(session, action, now=now, llm=llm)
            execution_results.append(exec_res)

            # Record candidate application in world model ONLY if action passed critic gate
            if not exec_res.dropped:
                app = Application(
                    goal_id=goal.id,
                    role_id=c.role_id,
                    stage=ApplicationStage.APPLIED,
                    applied_at=now,
                    notes=f"Applied via Pilot Cycle #{cycle_number}",
                )
                session.add(app)

        # Step 10: Failure Detection, Reflection, and Replay-Gated Adoption
        failure_sig = detect_strategy_failure(session, goal, active_strategy, now=now)
        adopt_res = None
        if failure_sig:
            refl = reflect(
                goal,
                active_strategy,
                failure_sig,
                [observation.model_dump(mode="json")],
                llm=llm,
            )
            adopt_res = evaluate_and_adopt(session, goal, refl, now=now, current_cycle=cycle)

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
            resolution_result=res_result,
            calibration_report=cal_report,
            failure_signal=failure_sig,
            adoption_result=adopt_res,
        )

    except Exception:
        session.rollback()
        raise
