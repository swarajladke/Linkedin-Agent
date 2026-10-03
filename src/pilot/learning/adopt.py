"""Counterfactual replay evaluation, strategy version adoption, and immutable rollback."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Cycle, Goal, Strategy, StrategyNote, StrategyNoteStatus
from pilot.learning.schemas import AdoptionResult, Reflection
from pilot.planner.replay import replay_cycle
from pilot.planner.score import score_actions


def evaluate_and_adopt(
    session: Session,
    goal: Goal,
    reflection: Reflection,
    *,
    lookback_cycles: int = 3,
    cooldown_cycles: int = 3,
    now: datetime | None = None,
    current_cycle: Cycle | None = None,
) -> AdoptionResult:
    """
    Evaluate proposed policy change against counterfactual replay before adopting.

    Adoption Pipeline:
    1. Check cooldown: must have at least M cycles under current strategy version.
    2. Check proposed changes: if empty, reject.
    3. Replay recent cycles under proposed policy.
    4. Reject if replay shows no material difference in action selection.
    5. On adoption: retire old version, create v+1 strategy with new policy,
       and record StrategyNote linking evidence, hypothesis, and replay diff.
    """
    adopt_time = now or datetime.now(UTC)

    # 1. Resolve current active strategy
    strat_stmt = (
        select(Strategy)
        .where(Strategy.goal_id == goal.id, Strategy.retired_at.is_(None))
        .order_by(Strategy.version.desc())
    )
    current_strategy = session.execute(strat_stmt).scalars().first()
    if not current_strategy:
        # Fallback to highest version
        strat_stmt_all = (
            select(Strategy).where(Strategy.goal_id == goal.id).order_by(Strategy.version.desc())
        )
        current_strategy = session.execute(strat_stmt_all).scalars().first()
        if not current_strategy:
            raise ValueError(f"Goal {goal.id} has no strategy versions to adapt.")

    old_v = current_strategy.version

    # 2. Check cooldown on current strategy
    cycle_count_stmt = select(Cycle).where(
        Cycle.goal_id == goal.id,
        Cycle.strategy_id == current_strategy.id,
    )
    strategy_cycles = session.execute(cycle_count_stmt).scalars().all()
    if len(strategy_cycles) < cooldown_cycles:
        return AdoptionResult(
            adopted=False,
            rejection_reason=(
                f"Cooldown active: strategy v{old_v} has run {len(strategy_cycles)} cycles "
                f"(requires at least {cooldown_cycles} cycles before another pivot)."
            ),
            old_version=old_v,
        )

    # 3. Must have proposed changes
    if not reflection.proposed_changes:
        return AdoptionResult(
            adopted=False,
            rejection_reason="Reflection contains no proposed policy changes.",
            old_version=old_v,
        )

    # 4. Synthesize proposed policy
    proposed_policy: dict[str, Any] = dict(current_strategy.policy or {})
    for change in reflection.proposed_changes:
        proposed_policy[change.parameter] = change.new_value

    # 5. Run counterfactual replay across the last N cycles
    recent_cycles_stmt = (
        select(Cycle)
        .where(Cycle.goal_id == goal.id)
        .order_by(Cycle.cycle_number.desc())
        .limit(lookback_cycles)
    )
    recent_cycles = session.execute(recent_cycles_stmt).scalars().all()

    replay_diffs: list[dict[str, Any]] = []
    total_added = 0
    total_removed = 0

    custom_scorer = lambda candidates, diagnosis, strategy_notes: score_actions(  # noqa: E731
        candidates, diagnosis, strategy_notes=strategy_notes, policy=proposed_policy
    )

    for c in recent_cycles:
        replay_res = replay_cycle(
            session,
            c.id,
            scorer=custom_scorer,
            min_fit=float(proposed_policy.get("fit_floor", 0.50)),
            max_actions=proposed_policy.get("max_applications_per_day"),
        )
        added_cnt = len(replay_res.added_action_ids)
        removed_cnt = len(replay_res.removed_action_ids)
        total_added += added_cnt
        total_removed += removed_cnt
        replay_diffs.append(
            {
                "cycle_number": c.cycle_number,
                "added_count": added_cnt,
                "removed_count": removed_cnt,
                "rationale": replay_res.rationale_diff,
            }
        )

    # 6. Reject if counterfactual selection is identical
    if total_added == 0 and total_removed == 0 and len(recent_cycles) > 0:
        return AdoptionResult(
            adopted=False,
            rejection_reason=(
                f"Replay rejected: proposed policy changes produced zero selection differences "
                f"across the last {len(recent_cycles)} cycles."
            ),
            old_version=old_v,
            replay_diff={"replayed_cycles": replay_diffs},
        )

    # 7. Adopt: retire old version, instantiate new version
    current_strategy.retired_at = adopt_time

    new_version = old_v + 1
    new_strategy = Strategy(
        goal_id=goal.id,
        version=new_version,
        parent_version_id=current_strategy.id,
        policy=proposed_policy,
        created_at=adopt_time,
    )
    session.add(new_strategy)
    session.flush()

    diff_summary = {
        "replayed_cycles_count": len(recent_cycles),
        "total_added": total_added,
        "total_removed": total_removed,
        "cycle_diffs": replay_diffs,
    }

    # 8. Record StrategyNote with triggering evidence, hypothesis, and replay diff
    note = StrategyNote(
        goal_id=goal.id,
        strategy_id=new_strategy.id,
        cycle_id=current_cycle.id
        if current_cycle
        else (recent_cycles[0].id if recent_cycles else None),
        hypothesis=reflection.hypothesis,
        reflection=reflection.explanation,
        evidence={
            "triggering_signal": reflection.signal.model_dump(mode="json"),
            "proposed_changes": [c.model_dump(mode="json") for c in reflection.proposed_changes],
            "replay_diff": diff_summary,
        },
        status=StrategyNoteStatus.ACTIVE,
        created_at=adopt_time,
    )
    session.add(note)
    session.flush()

    return AdoptionResult(
        adopted=True,
        old_version=old_v,
        new_version=new_version,
        new_strategy_id=new_strategy.id,
        replay_diff=diff_summary,
    )


def rollback_strategy(
    session: Session,
    goal: Goal,
    to_version: int,
    *,
    now: datetime | None = None,
) -> Strategy:
    """
    Rollback to the policy configuration of a historical version without mutating history.

    Guarantees:
    - Never mutates historical strategy rows.
    - Creates a new version (v+1) whose policy is an exact copy of `to_version`.
    - Retires the current active strategy.
    - Logs a StrategyNote for auditability.
    """
    rollback_time = now or datetime.now(UTC)

    # 1. Fetch target historical strategy
    target_stmt = select(Strategy).where(
        Strategy.goal_id == goal.id, Strategy.version == to_version
    )
    target_strategy = session.execute(target_stmt).scalars().first()
    if not target_strategy:
        raise ValueError(f"Strategy version {to_version} does not exist for goal {goal.id}.")

    # 2. Fetch current active strategy
    current_stmt = (
        select(Strategy)
        .where(Strategy.goal_id == goal.id, Strategy.retired_at.is_(None))
        .order_by(Strategy.version.desc())
    )
    current_strategy = session.execute(current_stmt).scalars().first()
    if not current_strategy:
        current_strategy = target_strategy

    if current_strategy.version == to_version:
        return current_strategy

    # 3. Retire current strategy
    current_strategy.retired_at = rollback_time

    # 4. Create new version copying the target policy
    new_version = current_strategy.version + 1
    new_strategy = Strategy(
        goal_id=goal.id,
        version=new_version,
        parent_version_id=current_strategy.id,
        policy=dict(target_strategy.policy or {}),
        created_at=rollback_time,
    )
    session.add(new_strategy)
    session.flush()

    # 5. Audit note
    note = StrategyNote(
        goal_id=goal.id,
        strategy_id=new_strategy.id,
        hypothesis=f"Rollback to policy from strategy version {to_version}.",
        reflection=(
            f"Audited rollback: strategy version {current_strategy.version} rolled back "
            f"to historical configuration of version {to_version}."
        ),
        evidence={
            "rollback_from_version": current_strategy.version,
            "rollback_to_version": to_version,
            "copied_policy": new_strategy.policy,
        },
        status=StrategyNoteStatus.ACTIVE,
        created_at=rollback_time,
    )
    session.add(note)
    session.flush()

    return new_strategy
