"""Pre-execution falsifiable prediction recording for Pilot Planner."""

from uuid import UUID

from sqlalchemy.orm import Session

from pilot.db.models import Action

# Explicit baseline prior for application conversion (Phase 5 will replace this seam with calibrated models)
DEFAULT_APPLICATION_CONVERSION_PRIOR = 0.70


def compute_prediction_prior(
    action: Action, default_prior: float = DEFAULT_APPLICATION_CONVERSION_PRIOR
) -> float:
    """
    Clean seam for probability estimation.
    In Phase 3, uses the explicitly configured or candidate-derived prior.
    In Phase 5, this will be swapped for calibrated probability deciles.
    """
    if action.predicted_probability is not None:
        return float(action.predicted_probability)
    return default_prior


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
    if prior is not None:
        action.predicted_probability = prior
    else:
        action.predicted_probability = compute_prediction_prior(action)

    if not action.predicted_outcome:
        action.predicted_outcome = (
            f"Action '{action.action_type}' on target '{action.target_id}' will succeed "
            f"with probability {action.predicted_probability:.2f} within 14 days."
        )

    if cycle_id is not None:
        action.cycle_id = cycle_id

    session.add(action)
    session.flush()
    return action
