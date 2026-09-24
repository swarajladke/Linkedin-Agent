"""Job intelligence schema: role sourcing columns and role assessments table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24 12:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add sourcing columns to roles table
    op.add_column(
        "roles",
        sa.Column(
            "source",
            sa.String(length=50),
            nullable=False,
            server_default="manual",
        ),
    )
    op.add_column(
        "roles",
        sa.Column(
            "external_id",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.add_column(
        "roles",
        sa.Column(
            "raw_posting",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "roles",
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.add_column(
        "roles",
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_unique_constraint(
        "uq_roles_source_external_id",
        "roles",
        ["source", "external_id"],
    )

    # 2. Create role_assessments table
    op.create_table(
        "role_assessments",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "role_id",
            sa.UUID(),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "goal_id",
            sa.UUID(),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "fit_score",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "fit_rationale",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "supporting_claim_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "skill_gaps",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "company_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "recommended_action",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "assessed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "assessor_version",
            sa.String(length=50),
            nullable=False,
        ),
        sa.CheckConstraint(
            "fit_score >= 0.0 AND fit_score <= 1.0",
            name="chk_role_assessments_fit_score",
        ),
        sa.UniqueConstraint(
            "role_id",
            "goal_id",
            "assessor_version",
            name="uq_role_assessments_role_goal_version",
        ),
    )
    op.create_index(
        "ix_role_assessments_goal_fit",
        "role_assessments",
        ["goal_id", "fit_score"],
    )


def downgrade() -> None:
    # 1. Drop role_assessments table and its index
    op.drop_index("ix_role_assessments_goal_fit", table_name="role_assessments")
    op.drop_table("role_assessments")

    # 2. Revert roles sourcing columns and constraint
    op.drop_constraint("uq_roles_source_external_id", "roles", type_="unique")
    op.drop_column("roles", "last_seen_at")
    op.drop_column("roles", "first_seen_at")
    op.drop_column("roles", "raw_posting")
    op.drop_column("roles", "external_id")
    op.drop_column("roles", "source")
