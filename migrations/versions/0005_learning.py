"""Learning schema: strategy policy, note evidence and cycle link, and action horizon.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-03 12:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add policy column to strategies
    op.add_column(
        "strategies",
        sa.Column(
            "policy",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    # Backfill v1 policy for existing strategies
    op.execute(
        """
        UPDATE strategies
        SET policy = '{"fit_floor": 0.50, "elevated_fit_floor": 0.75, "conversion_prior_factor": 0.40, "default_conversion_prior": 0.70, "horizon_days": 14}'::jsonb
        WHERE policy = '{}'::jsonb OR policy IS NULL
        """
    )

    # 2. Add evidence and cycle_id columns to strategy_notes
    op.add_column(
        "strategy_notes",
        sa.Column(
            "evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "strategy_notes",
        sa.Column(
            "cycle_id",
            sa.UUID(),
            sa.ForeignKey("cycles.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_strategy_notes_cycle_id",
        "strategy_notes",
        ["cycle_id"],
    )

    # 3. Add horizon_days column to actions
    op.add_column(
        "actions",
        sa.Column(
            "horizon_days",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("14"),
        ),
    )


def downgrade() -> None:
    # 1. Drop horizon_days from actions
    op.drop_column("actions", "horizon_days")

    # 2. Drop cycle_id index and columns from strategy_notes
    op.drop_index("ix_strategy_notes_cycle_id", table_name="strategy_notes")
    op.drop_column("strategy_notes", "cycle_id")
    op.drop_column("strategy_notes", "evidence")

    # 3. Drop policy from strategies
    op.drop_column("strategies", "policy")
