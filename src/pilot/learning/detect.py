"""Deterministic strategy failure detection based on statistical and diagnostic signals."""

import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import (
    Action,
    ActionOutcome,
    Cycle,
    Goal,
    Strategy,
    StrategyNote,
    StrategyNoteStatus,
)
from pilot.learning.outcomes import is_critic_dropped
from pilot.learning.schemas import FailureReasonKind, FailureSignal


def detect_strategy_failure(
    session: Session,
    goal: Goal,
    strategy: Strategy,
    *,
    min_sample_size: int = 5,
    consecutive_starved_threshold: int = 3,
    critic_drop_threshold: float = 0.30,
    calibration_drift_threshold: float = 0.15,
    now: datetime | None = None,
) -> FailureSignal | None:
    """
    Evaluate deterministic failure signals for the current strategy version.

    Checks (in priority order):
    1. Sub-goal deadline missed.
    2. Refuted strategy hypothesis.
    3. Consecutive starved cycles for the same funnel stage.
    4. Critic drop rate exceeding threshold.
    5. Observed conversion rate breaching Beta posterior credible interval.
    6. Calibration drift (recent Brier score materially worse than baseline).

    Guarantees:
    - Below minimum sample size, statistical triggers return None (no pivoting on noise).
    - Pure, deterministic evaluation.
    """
    check_time = now or datetime.now(UTC)

    # 1. Check missed deadline on sub-goals
    if goal.sub_goals and isinstance(goal.sub_goals, list):
        for sg in goal.sub_goals:
            if isinstance(sg, dict):
                deadline_str = sg.get("deadline") or sg.get("target_date")
                status = sg.get("status")
                if deadline_str and status not in ("completed", "achieved", "done"):
                    try:
                        dl = datetime.fromisoformat(deadline_str)
                        if dl.tzinfo is None:
                            dl = dl.replace(tzinfo=UTC)
                        if check_time > dl:
                            return FailureSignal(
                                kind=FailureReasonKind.MISSED_DEADLINE,
                                summary=f"Sub-goal '{sg.get('title', 'milestone')}' missed deadline {deadline_str}.",
                                details={"sub_goal": sg, "check_time": check_time.isoformat()},
                            )
                    except Exception:
                        pass

    # 2. Check refuted hypothesis on strategy notes
    refuted_stmt = (
        select(StrategyNote)
        .where(
            StrategyNote.strategy_id == strategy.id,
            StrategyNote.status == StrategyNoteStatus.REFUTED,
        )
        .order_by(StrategyNote.created_at.desc())
    )
    refuted_note = session.execute(refuted_stmt).scalars().first()
    if refuted_note:
        return FailureSignal(
            kind=FailureReasonKind.REFUTED_HYPOTHESIS,
            summary=f"Strategy hypothesis was refuted: '{refuted_note.hypothesis}'.",
            details={
                "note_id": str(refuted_note.id),
                "hypothesis": refuted_note.hypothesis,
                "evidence": refuted_note.evidence,
            },
        )

    # 3. Check K consecutive starved cycles for the same stage
    cycle_stmt = (
        select(Cycle)
        .where(Cycle.goal_id == goal.id, Cycle.strategy_id == strategy.id)
        .order_by(Cycle.cycle_number.desc())
        .limit(consecutive_starved_threshold)
    )
    recent_cycles = session.execute(cycle_stmt).scalars().all()
    if len(recent_cycles) >= consecutive_starved_threshold:
        starved_stages: list[str] = []
        all_starved = True
        for c in recent_cycles:
            diag = c.diagnosis if isinstance(c.diagnosis, dict) else {}
            if diag.get("category") == "starved" and diag.get("starved_stage"):
                starved_stages.append(str(diag.get("starved_stage")))
            else:
                all_starved = False
                break

        if all_starved and len(set(starved_stages)) == 1:
            stage_name = starved_stages[0]
            return FailureSignal(
                kind=FailureReasonKind.CONSECUTIVE_STARVED,
                stage=stage_name,
                consecutive_cycles=consecutive_starved_threshold,
                summary=(
                    f"Stage '{stage_name}' has been starved for {consecutive_starved_threshold} "
                    f"consecutive cycles."
                ),
                details={"consecutive_cycles": len(recent_cycles), "stage": stage_name},
            )

    # 4. Check critic drop spike on recent actions
    actions_stmt = (
        select(Action)
        .where(Action.strategy_id == strategy.id)
        .order_by(Action.created_at.desc())
        .limit(10)
    )
    recent_actions = session.execute(actions_stmt).scalars().all()
    if len(recent_actions) >= 5:
        dropped_count = sum(1 for a in recent_actions if is_critic_dropped(a))
        drop_rate = dropped_count / len(recent_actions)
        if drop_rate > critic_drop_threshold:
            return FailureSignal(
                kind=FailureReasonKind.CRITIC_DROP_SPIKE,
                metric_value=round(drop_rate, 4),
                threshold=critic_drop_threshold,
                summary=(
                    f"Critic drop rate is {drop_rate:.1%}, exceeding threshold "
                    f"{critic_drop_threshold:.1%} over last {len(recent_actions)} actions."
                ),
                details={"dropped_count": dropped_count, "total_checked": len(recent_actions)},
            )

    # 5. Fetch resolved action outcomes for statistical signals
    outcomes_stmt = (
        select(
            Action.predicted_probability, ActionOutcome.binary_success, ActionOutcome.brier_score
        )
        .join(ActionOutcome, ActionOutcome.action_id == Action.id)
        .where(Action.strategy_id == strategy.id)
        .order_by(Action.created_at.asc())
    )
    records = session.execute(outcomes_stmt).all()
    sample_size = len(records)

    # Never pivot on noise: sample floor for statistical triggers
    if sample_size < min_sample_size:
        return None

    # 5a. Credible interval breach on stage conversion rate
    policy: dict[str, Any] = strategy.policy or {}
    p0 = float(policy.get("default_conversion_prior", 0.70))
    v0 = float(policy.get("prior_strength", 10.0))

    successes = sum(1 for r in records if bool(r[1]))
    n = sample_size
    obs_rate = successes / n

    alpha_0 = v0 * p0
    beta_0 = v0 * (1.0 - p0)
    alpha_post = alpha_0 + successes
    beta_post = beta_0 + (n - successes)

    post_mean = alpha_post / (alpha_post + beta_post)
    post_var = (alpha_post * beta_post) / (
        ((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1.0)
    )
    post_sd = math.sqrt(post_var)
    ci_lower = max(0.0, post_mean - 1.96 * post_sd)

    if obs_rate < ci_lower:
        return FailureSignal(
            kind=FailureReasonKind.CREDIBLE_INTERVAL_BREACHED,
            metric_value=round(obs_rate, 4),
            threshold=round(ci_lower, 4),
            summary=(
                f"Observed conversion rate ({obs_rate:.1%}) breached lower bound of "
                f"Beta credible interval ({ci_lower:.1%}) for prior {p0:.1%}."
            ),
            details={
                "observed_rate": obs_rate,
                "ci_lower": ci_lower,
                "posterior_mean": post_mean,
                "successes": successes,
                "total": n,
            },
        )

    # 5b. Calibration drift (recent Brier score worse than baseline)
    if sample_size >= 10:
        half = sample_size // 2
        early_brier = sum(float(r[2]) for r in records[:half]) / half
        recent_brier = sum(float(r[2]) for r in records[half:]) / (sample_size - half)

        if (recent_brier - early_brier) > calibration_drift_threshold:
            return FailureSignal(
                kind=FailureReasonKind.CALIBRATION_DRIFT,
                metric_value=round(recent_brier, 4),
                threshold=round(early_brier + calibration_drift_threshold, 4),
                summary=(
                    f"Rolling Brier score drifted from {early_brier:.4f} to {recent_brier:.4f} "
                    f"(delta {recent_brier - early_brier:.4f} > {calibration_drift_threshold:.4f})."
                ),
                details={
                    "early_brier": early_brier,
                    "recent_brier": recent_brier,
                    "sample_size": sample_size,
                },
            )

    return None
