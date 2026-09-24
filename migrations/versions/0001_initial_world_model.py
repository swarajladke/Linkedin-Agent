"""Initial world model schema, triggers, and extensions.

Revision ID: 0001
Revises: None
Create Date: 2026-09-24 07:00:00.000000+00:00

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from pilot.constants import EMBEDDING_DIM

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Enums
goal_status_enum = postgresql.ENUM(
    "active", "paused", "achieved", "abandoned", name="goal_status", create_type=False
)
application_stage_enum = postgresql.ENUM(
    "discovered",
    "preparing",
    "applied",
    "screening",
    "technical",
    "final",
    "offer",
    "rejected",
    "withdrawn",
    name="application_stage",
    create_type=False,
)
role_status_enum = postgresql.ENUM(
    "open", "applied", "closed", name="role_status", create_type=False
)
strategy_note_status_enum = postgresql.ENUM(
    "active", "refuted", "graduated", name="strategy_note_status", create_type=False
)


def upgrade() -> None:
    # 1. Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # 2. Postgres ENUM Types
    goal_status_enum.create(op.get_bind(), checkfirst=True)
    application_stage_enum.create(op.get_bind(), checkfirst=True)
    role_status_enum.create(op.get_bind(), checkfirst=True)
    strategy_note_status_enum.create(op.get_bind(), checkfirst=True)

    # 3. updated_at Trigger Function
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # 4. Tables
    # users
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("github_username", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.execute(
        """
        CREATE TRIGGER trg_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
        """
    )

    # goals
    op.create_table(
        "goals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("objective_text", sa.Text(), nullable=False),
        sa.Column("constraints_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("target_spec", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("success_criteria", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("sub_goals", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", goal_status_enum, server_default="active", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_goals_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_goals")),
    )
    op.execute(
        """
        CREATE TRIGGER trg_goals_updated_at
        BEFORE UPDATE ON goals
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
        """
    )

    # strategies
    op.create_table(
        "strategies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("parent_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_strategies_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["strategies.id"],
            name=op.f("fk_strategies_parent_version_id_strategies"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_strategies")),
        sa.UniqueConstraint("goal_id", "version", name="uq_strategies_goal_id_version"),
    )

    # companies
    op.create_table(
        "companies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=True),
        sa.Column("careers_url", sa.Text(), nullable=True),
        sa.Column("headquarters", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_companies")),
    )
    op.create_index(op.f("ix_companies_name"), "companies", ["name"], unique=False)

    # people
    op.create_table(
        "people",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("role_title", sa.String(length=255), nullable=True),
        sa.Column("public_profile_url", sa.Text(), nullable=True),
        sa.Column("connection_degree", sa.String(length=20), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_people_company_id_companies"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_people")),
    )

    # roles
    op.create_table(
        "roles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("location_type", sa.String(length=50), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("posting_url", sa.Text(), nullable=True),
        sa.Column("requirements_summary", sa.Text(), nullable=True),
        sa.Column("status", role_status_enum, server_default="open", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name=op.f("fk_roles_company_id_companies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roles")),
    )
    op.create_index("ix_roles_company_id_status", "roles", ["company_id", "status"], unique=False)

    # applications
    op.create_table(
        "applications",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", application_stage_enum, server_default="discovered", nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("portal_url", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_applications_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["role_id"],
            ["roles.id"],
            name=op.f("fk_applications_role_id_roles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_applications")),
    )
    op.create_index(
        "ix_applications_goal_id_stage", "applications", ["goal_id", "stage"], unique=False
    )
    op.execute(
        """
        CREATE TRIGGER trg_applications_updated_at
        BEFORE UPDATE ON applications
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
        """
    )

    # conversations
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("application_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("channel", sa.String(length=50), nullable=False),
        sa.Column("thread_summary", sa.Text(), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["applications.id"],
            name=op.f("fk_conversations_application_id_applications"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_conversations_person_id_people"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )

    # actions
    op.create_table(
        "actions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("target_type", sa.String(length=50), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("predicted_outcome", sa.Text(), nullable=False),
        sa.Column("predicted_probability", sa.Float(), nullable=False),
        sa.Column("cost", sa.Numeric(precision=10, scale=4), server_default="0.0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_outcome", sa.Text(), nullable=True),
        sa.Column("outcome_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "predicted_probability >= 0.0 AND predicted_probability <= 1.0",
            name="ck_actions_predicted_probability_bounds",
        ),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_actions_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["strategies.id"],
            name=op.f("fk_actions_strategy_id_strategies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_actions")),
    )
    op.create_index(
        "ix_actions_goal_id_created_at", "actions", ["goal_id", "created_at"], unique=False
    )
    op.create_index("ix_actions_strategy_id", "actions", ["strategy_id"], unique=False)

    # action_outcomes
    op.create_table(
        "action_outcomes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("action_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("binary_success", sa.Boolean(), nullable=False),
        sa.Column("actual_outcome_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=False),
        sa.Column("diagnosis", sa.Text(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["action_id"],
            ["actions.id"],
            name=op.f("fk_action_outcomes_action_id_actions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action_outcomes")),
        sa.UniqueConstraint("action_id", name=op.f("uq_action_outcomes_action_id")),
    )

    # 5. Brier Score Trigger Function and Trigger
    op.execute(
        """
        CREATE OR REPLACE FUNCTION compute_action_outcome_brier_score()
        RETURNS TRIGGER AS $$
        DECLARE
            act_prob DOUBLE PRECISION;
        BEGIN
            SELECT predicted_probability INTO act_prob
            FROM actions
            WHERE id = NEW.action_id;

            IF act_prob IS NULL THEN
                RAISE EXCEPTION 'Parent action % has no predicted_probability; Brier score cannot be computed', NEW.action_id;
            END IF;

            NEW.brier_score := POWER(act_prob - (CASE WHEN NEW.binary_success THEN 1.0 ELSE 0.0 END), 2);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_action_outcomes_brier_score
        BEFORE INSERT OR UPDATE ON action_outcomes
        FOR EACH ROW
        EXECUTE FUNCTION compute_action_outcome_brier_score();
        """
    )

    # evidence_claims
    op.create_table(
        "evidence_claims",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_excerpt", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "verified_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_evidence_claims_confidence_bounds",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_claims")),
        sa.UniqueConstraint("entity_id", "content_hash", name="uq_evidence_claims_entity_hash"),
    )
    op.create_index(
        "ix_evidence_claims_entity_type_entity_id",
        "evidence_claims",
        ["entity_type", "entity_id"],
        unique=False,
    )

    # predictions (funnel-level only)
    op.create_table(
        "predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_statement", sa.Text(), nullable=False),
        sa.Column("predicted_probability", sa.Float(), nullable=False),
        sa.Column("resolution_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_resolved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("outcome_truth", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "predicted_probability >= 0.0 AND predicted_probability <= 1.0",
            name="ck_predictions_predicted_probability_bounds",
        ),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_predictions_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )

    # calibration
    op.create_table(
        "calibration",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("window_name", sa.String(length=50), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("mean_brier_score", sa.Float(), nullable=False),
        sa.Column("calibration_buckets", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "calculated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["strategies.id"],
            name=op.f("fk_calibration_strategy_id_strategies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_calibration")),
    )
    op.create_index("ix_calibration_strategy_id", "calibration", ["strategy_id"], unique=False)

    # strategy_notes
    op.create_table(
        "strategy_notes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("strategy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("reflection", sa.Text(), nullable=True),
        sa.Column("status", strategy_note_status_enum, server_default="active", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_strategy_notes_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["strategies.id"],
            name=op.f("fk_strategy_notes_strategy_id_strategies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_strategy_notes")),
    )
    op.create_index("ix_strategy_notes_goal_id", "strategy_notes", ["goal_id"], unique=False)
    op.create_index(
        "ix_strategy_notes_strategy_id", "strategy_notes", ["strategy_id"], unique=False
    )

    # escalations
    op.create_table(
        "escalations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("goal_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("resolved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["action_id"],
            ["actions.id"],
            name=op.f("fk_escalations_action_id_actions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["goal_id"],
            ["goals.id"],
            name=op.f("fk_escalations_goal_id_goals"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_escalations")),
    )


def downgrade() -> None:
    # 1. Drop tables in reverse topological order
    op.drop_table("escalations")
    op.drop_index("ix_strategy_notes_strategy_id", table_name="strategy_notes")
    op.drop_index("ix_strategy_notes_goal_id", table_name="strategy_notes")
    op.drop_table("strategy_notes")
    op.drop_index("ix_calibration_strategy_id", table_name="calibration")
    op.drop_table("calibration")
    op.drop_table("predictions")
    op.drop_index("ix_evidence_claims_entity_type_entity_id", table_name="evidence_claims")
    op.drop_table("evidence_claims")

    # Drop Brier score trigger and function
    op.execute("DROP TRIGGER IF EXISTS trg_action_outcomes_brier_score ON action_outcomes;")
    op.execute("DROP FUNCTION IF EXISTS compute_action_outcome_brier_score();")

    op.drop_table("action_outcomes")
    op.drop_index("ix_actions_strategy_id", table_name="actions")
    op.drop_index("ix_actions_goal_id_created_at", table_name="actions")
    op.drop_table("actions")
    op.drop_table("conversations")

    # Drop applications trigger and table
    op.execute("DROP TRIGGER IF EXISTS trg_applications_updated_at ON applications;")
    op.drop_index("ix_applications_goal_id_stage", table_name="applications")
    op.drop_table("applications")

    op.drop_index("ix_roles_company_id_status", table_name="roles")
    op.drop_table("roles")
    op.drop_table("people")
    op.drop_index(op.f("ix_companies_name"), table_name="companies")
    op.drop_table("companies")
    op.drop_table("strategies")

    # Drop goals trigger and table
    op.execute("DROP TRIGGER IF EXISTS trg_goals_updated_at ON goals;")
    op.drop_table("goals")

    # Drop users trigger and table
    op.execute("DROP TRIGGER IF EXISTS trg_users_updated_at ON users;")
    op.drop_table("users")

    # Drop updated_at function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # 2. Drop Enums
    bind = op.get_bind()
    strategy_note_status_enum.drop(bind, checkfirst=True)
    role_status_enum.drop(bind, checkfirst=True)
    application_stage_enum.drop(bind, checkfirst=True)
    goal_status_enum.drop(bind, checkfirst=True)

    # 3. Drop Extensions
    op.execute("DROP EXTENSION IF EXISTS pgcrypto;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
