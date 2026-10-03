"""Pre-execution falsifiable prediction recording for Pilot Planner."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, object_session

from pilot.db.models import Action, ActionOutcome, Strategy
from pilot.learning.calibration import calibrated_probability

# Baseline default prior for application conversion
DEFAULT_APPLICATION_CONVERSION_PRIOR = 0.70


def compute_prediction_prior(
    action: Action,
    default_prior: float = DEFAULT_APPLICATION_CONVERSION_PRIOR,
    session: Session | None = None,
    raw_prior: float | None = None,
) -> float:
    """
    Clean seam for Bayesian calibrated probability estimation.

    In Phase 3, this used an uncalibrated prior.
    In Phase 5, this plugs in calibrated Bayesian shrinkage using historical outcomes
    and strategy policy assumptions.
    """
    sess = session or object_session(action)

    # 1. Determine raw prior from passed raw_prior, action, or policy
    policy: dict = {}
    if action.strategy and action.strategy.policy:
        policy = action.strategy.policy
    elif action.strategy_id and sess:
        strat = sess.get(Strategy, action.strategy_id)
        if strat and strat.policy:
            policy = strat.policy

    base_p0 = float(policy.get("default_conversion_prior", default_prior))
    if raw_prior is not None:
        p0 = float(raw_prior)
    elif action.predicted_probability is not None:
        p0 = float(action.predicted_probability)
    else:
        p0 = base_p0

    # 2. Extract historical outcomes for this goal and action type
    history: list[bool] = []
    if sess is not None and action.goal_id:
        try:
            stmt = (
                select(ActionOutcome.binary_success)
                .join(Action, Action.id == ActionOutcome.action_id)
                .where(
                    Action.goal_id == action.goal_id,
                    Action.action_type == action.action_type,
                )
            )
            history = list(sess.scalars(stmt).all())
        except Exception:
            history = []

    # 3. Compute shrinkage
    prior_strength = float(policy.get("prior_strength", 10.0))
    calibrated = calibrated_probability(
        raw=p0,
        history=history,
        prior_strength=prior_strength,
        default_prior=base_p0,
    )
    return calibrated


def record_prediction(
    session: Session,
    action: Action,
    *,
    cycle_id: UUID | None = None,
    prior: float | None = None,
) -> Action:
    """
    Persist the action row with its falsifiable prediction BEFORE execution.

    Structural requirement:
    - Every action must have non-empty reason, predicted_outcome, and predicted_probability in [0, 1].
    - Flushes to ensure DB state before execute() can run.
    """
    # 1. Resolve policy horizon days
    horizon = 14
    if action.strategy and action.strategy.policy:
        horizon = int(action.strategy.policy.get("horizon_days", 14))
    elif action.strategy_id:
        try:
            strat = session.get(Strategy, action.strategy_id)
            if strat and strat.policy:
                horizon = int(strat.policy.get("horizon_days", 14))
        except Exception:
            pass
    action.horizon_days = horizon

    # 2. Compute calibrated probability through the prediction seam
    action.predicted_probability = compute_prediction_prior(
        action,
        session=session,
        raw_prior=prior,
    )

    if not action.predicted_outcome:
        action.predicted_outcome = (
            f"Action '{action.action_type}' on target '{action.target_id}' will succeed "
            f"with probability {action.predicted_probability:.2f} within {action.horizon_days} days."
        )

    if cycle_id is not None:
        action.cycle_id = cycle_id

    session.add(action)
    session.flush()
    return action
