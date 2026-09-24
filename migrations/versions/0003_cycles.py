"""Planner cycle schema: cycles table and actions cycle_id foreign key.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-24 14:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create cycles table
    op.create_table(
        "cycles",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "goal_id",
            sa.UUID(),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cycle_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "strategy_id",
            sa.UUID(),
            sa.ForeignKey("strategies.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "observation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "diagnosis",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "actions_proposed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "actions_selected",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.UniqueConstraint(
            "goal_id",
            "cycle_number",
            name="uq_cycles_goal_id_cycle_number",
        ),
    )

    # 2. Add cycle_id to actions table
    op.add_column(
        "actions",
        sa.Column(
            "cycle_id",
            sa.UUID(),
            sa.ForeignKey("cycles.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_actions_cycle_id",
        "actions",
        ["cycle_id"],
    )


def downgrade() -> None:
    # 1. Drop cycle_id from actions
    op.drop_index("ix_actions_cycle_id", table_name="actions")
    op.drop_column("actions", "cycle_id")

    # 2. Drop cycles table
    op.drop_table("cycles")
