"""Critic schema: writing_samples and critic_reviews tables.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-25 10:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create writing_samples table
    op.create_table(
        "writing_samples",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_url",
            sa.String(length=512),
            nullable=True,
        ),
        sa.Column(
            "content_type",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "content_hash",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "raw_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "spans",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "word_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "voice_profile",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "user_id",
            "content_hash",
            name="uq_writing_samples_user_id_content_hash",
        ),
    )
    op.create_index(
        "ix_writing_samples_user_id",
        "writing_samples",
        ["user_id"],
    )

    # 2. Create critic_reviews table
    op.create_table(
        "critic_reviews",
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "action_id",
            sa.UUID(),
            sa.ForeignKey("actions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "artifact_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "grounding_passed",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "voice_passed",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "factual_passed",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "verdict",
            sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            "failures",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "reviewed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "action_id",
            "attempt",
            name="uq_critic_reviews_action_id_attempt",
        ),
    )
    op.create_index(
        "ix_critic_reviews_action_id",
        "critic_reviews",
        ["action_id"],
    )


def downgrade() -> None:
    # 1. Drop critic_reviews table
    op.drop_index("ix_critic_reviews_action_id", table_name="critic_reviews")
    op.drop_table("critic_reviews")

    # 2. Drop writing_samples table
    op.drop_index("ix_writing_samples_user_id", table_name="writing_samples")
    op.drop_table("writing_samples")
