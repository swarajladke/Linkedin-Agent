"""Deterministic probability calibration and Bayesian shrinkage estimation."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Action, ActionOutcome, Calibration
from pilot.learning.schemas import (
    CalibrationReport,
    DecileBucket,
    MurphyDecomposition,
)


def compute_calibration(
    session: Session,
    strategy_id: UUID,
    *,
    now: datetime | None = None,
) -> CalibrationReport:
    """
    Compute deterministic probability calibration metrics across all resolved action outcomes.

    Calculates:
    - Reliability by probability decile (predicted mean vs. observed rate, count per bucket)
    - Mean Brier score and Murphy (1973) decomposition: BS = Reliability - Resolution + Uncertainty
    - Expected Calibration Error (ECE)
    - Skill score against a climatological base-rate predictor
    - Persists report to calibration table idempotently.
    """
    calc_time = now or datetime.now(UTC)

    # 1. Fetch all resolved action outcomes for this strategy
    stmt = (
        select(Action.predicted_probability, ActionOutcome.binary_success)
        .join(ActionOutcome, ActionOutcome.action_id == Action.id)
        .where(Action.strategy_id == strategy_id)
        .order_by(Action.created_at.asc())
    )
    records = session.execute(stmt).all()

    sample_size = len(records)
    if sample_size == 0:
        # Empty fallback report
        deciles = [
            DecileBucket(
                decile=i + 1,
                bin_lower=round(i * 0.1, 1),
                bin_upper=round((i + 1) * 0.1, 1),
                count=0,
                mean_predicted=round(i * 0.1 + 0.05, 2),
                observed_rate=0.0,
            )
            for i in range(10)
        ]
        murphy = MurphyDecomposition(
            reliability=0.0,
            resolution=0.0,
            uncertainty=0.0,
            brier_score=0.0,
        )
        report = CalibrationReport(
            strategy_id=strategy_id,
            sample_size=0,
            mean_brier_score=0.0,
            ece=0.0,
            brier_skill_score=0.0,
            murphy_decomposition=murphy,
            deciles=deciles,
            base_rate=0.0,
            beats_base_rate=False,
            summary="No resolved outcomes available for strategy.",
        )
        return report

    # 2. Extract predictions and binary outcomes
    preds = [float(r[0]) for r in records]
    targets = [1.0 if bool(r[1]) else 0.0 for r in records]
    base_rate = sum(targets) / sample_size

    # 3. Bin predictions into 10 deciles [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bins_preds: list[list[float]] = [[] for _ in range(10)]
    bins_targets: list[list[float]] = [[] for _ in range(10)]

    for p, o in zip(preds, targets, strict=True):
        idx = min(9, int(p * 10.0))
        bins_preds[idx].append(p)
        bins_targets[idx].append(o)

    decile_buckets: list[DecileBucket] = []
    ece_acc = 0.0

    # Components for Murphy decomposition
    rel_acc = 0.0
    res_acc = 0.0

    for i in range(10):
        b_preds = bins_preds[i]
        b_targets = bins_targets[i]
        n_b = len(b_preds)
        bin_low = round(i * 0.1, 1)
        bin_up = round((i + 1) * 0.1, 1)

        if n_b > 0:
            p_bar = sum(b_preds) / n_b
            o_bar = sum(b_targets) / n_b
            ece_acc += (n_b / sample_size) * abs(p_bar - o_bar)
            rel_acc += (n_b / sample_size) * ((p_bar - o_bar) ** 2)
            res_acc += (n_b / sample_size) * ((o_bar - base_rate) ** 2)
        else:
            p_bar = round(bin_low + 0.05, 2)
            o_bar = 0.0

        decile_buckets.append(
            DecileBucket(
                decile=i + 1,
                bin_lower=bin_low,
                bin_upper=bin_up,
                count=n_b,
                mean_predicted=round(p_bar, 4),
                observed_rate=round(o_bar, 4),
            )
        )

    # 4. Murphy Decomposition
    unc = base_rate * (1.0 - base_rate)
    murphy_bs = rel_acc - res_acc + unc
    raw_brier = sum((p - o) ** 2 for p, o in zip(preds, targets, strict=True)) / sample_size

    # Ensure Murphy identity: BS = REL - RES + UNC
    murphy = MurphyDecomposition(
        reliability=round(rel_acc, 6),
        resolution=round(res_acc, 6),
        uncertainty=round(unc, 6),
        brier_score=round(murphy_bs, 6),
    )

    # 5. Brier Skill Score vs Climatological Base-Rate Predictor
    if unc > 1e-6:
        brier_skill = (res_acc - rel_acc) / unc
    else:
        brier_skill = 0.0 if raw_brier < 1e-4 else -1.0

    beats_base = brier_skill > 0.0
    summary = (
        f"Calibrated over {sample_size} outcomes. Mean Brier: {raw_brier:.4f}, ECE: {ece_acc:.4f}. "
        f"Skill vs base-rate: {brier_skill:.3f} ({'BEATS base-rate' if beats_base else 'FAILS to beat base-rate'})."
    )

    report = CalibrationReport(
        strategy_id=strategy_id,
        sample_size=sample_size,
        mean_brier_score=round(raw_brier, 4),
        ece=round(ece_acc, 4),
        brier_skill_score=round(brier_skill, 4),
        murphy_decomposition=murphy,
        deciles=decile_buckets,
        base_rate=round(base_rate, 4),
        beats_base_rate=beats_base,
        summary=summary,
    )

    # 6. Idempotently persist to calibration table
    existing_cal = (
        session.execute(
            select(Calibration).where(
                Calibration.strategy_id == strategy_id,
                Calibration.window_name == "cumulative",
            )
        )
        .scalars()
        .first()
    )

    if existing_cal:
        existing_cal.sample_size = sample_size
        existing_cal.mean_brier_score = report.mean_brier_score
        existing_cal.calibration_buckets = report.model_dump(mode="json")
        existing_cal.calculated_at = calc_time
    else:
        new_cal = Calibration(
            strategy_id=strategy_id,
            window_name="cumulative",
            sample_size=sample_size,
            mean_brier_score=report.mean_brier_score,
            calibration_buckets=report.model_dump(mode="json"),
            calculated_at=calc_time,
        )
        session.add(new_cal)

    session.flush()
    return report


def calibrated_probability(
    raw: float,
    history: list[bool] | list[ActionOutcome] | None,
    *,
    prior_strength: float = 10.0,
    default_prior: float = 0.70,
) -> float:
    """
    Bayesian shrinkage probability estimator plugging into the predict prior seam.

    Uses Beta prior updated with observed successes and failures:
    - Zero data returns the prior.
    - As n grows, converges smoothly to the observed empirical rate.
    - Monotonically increasing in observed successes.
    - Bounded safely away from 0 and 1.
    """
    p0 = raw if raw is not None else default_prior
    p0 = max(0.01, min(0.99, float(p0)))

    if not history:
        return round(p0, 4)

    # Extract successes from bool list or ActionOutcome list
    successes = 0
    total = len(history)

    for item in history:
        if isinstance(item, bool):
            if item:
                successes += 1
        elif isinstance(item, ActionOutcome):
            if item.binary_success:
                successes += 1
        elif isinstance(item, dict):
            if item.get("binary_success"):
                successes += 1

    # Beta shrinkage: (v0 * p0 + k) / (v0 + n)
    v0 = max(1.0, float(prior_strength))
    p_hat = (v0 * p0 + float(successes)) / (v0 + float(total))
    clamped = max(0.01, min(0.99, p_hat))
    return round(clamped, 4)
