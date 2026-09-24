"""Comprehensive test for Alembic migrations, schema fidelity, constraints, and triggers."""

import uuid

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from pilot.config import get_settings
from pilot.db.session import Base


def get_alembic_config() -> Config:
    """Return Alembic config configured with runtime database url."""
    settings = get_settings()
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url.get_secret_value())
    return alembic_cfg


def test_full_migration_lifecycle_and_constraints(db_engine):
    """Test upgrade -> zero-diff autogenerate -> constraints -> triggers -> downgrade."""
    alembic_cfg = get_alembic_config()

    # 1. Run upgrade to head
    command.upgrade(alembic_cfg, "head")

    with db_engine.connect() as connection:
        # 2. Assert zero schema diff between Base.metadata and live DB
        migration_context = MigrationContext.configure(connection)
        diff = compare_metadata(migration_context, Base.metadata)
        assert diff == [], f"Autogenerate diff is not empty: {diff}"

        # 3. Test CheckConstraint: action predicted_probability in [0, 1]
        trans = connection.begin()
        # Seed user, goal, strategy
        user_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        strategy_id = uuid.uuid4()

        connection.execute(
            text(
                """
                INSERT INTO users (id, email, full_name)
                VALUES (:user_id, 'candidate@example.com', 'Alex Candidate')
                """
            ),
            {"user_id": user_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO goals (id, user_id, objective_text, constraints_json, target_spec, success_criteria, sub_goals, deadline)
                VALUES (:goal_id, :user_id, 'Land Applied AI role', '{}', '{}', '{}', '[]', NOW() + INTERVAL '30 days')
                """
            ),
            {"goal_id": goal_id, "user_id": user_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO strategies (id, goal_id, version)
                VALUES (:strategy_id, :goal_id, 1)
                """
            ),
            {"strategy_id": strategy_id, "goal_id": goal_id},
        )

        # Attempt invalid probability = 1.5 -> must fail CheckConstraint
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    """
                    INSERT INTO actions (
                        id, goal_id, strategy_id, action_type, target_type,
                        reason, predicted_outcome, predicted_probability, strategy_version
                    ) VALUES (
                        gen_random_uuid(), :goal_id, :strategy_id, 'test_action', 'role',
                        'test reason', 'test predicted outcome', 1.5, 'v1'
                    )
                    """
                ),
                {"goal_id": goal_id, "strategy_id": strategy_id},
            )
        trans.rollback()

        # 4. Test UniqueConstraint on evidence_claims (entity_id, content_hash)
        trans = connection.begin()
        claim_entity_id = uuid.uuid4()
        test_hash = "a" * 64
        connection.execute(
            text(
                """
                INSERT INTO evidence_claims (
                    id, entity_type, entity_id, claim, source, source_url, source_excerpt, content_hash, confidence
                ) VALUES (
                    gen_random_uuid(), 'user', :entity_id, 'Claim 1', 'resume', '/path/resume.pdf', 'excerpt text', :content_hash, 0.95
                )
                """
            ),
            {"entity_id": claim_entity_id, "content_hash": test_hash},
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    """
                    INSERT INTO evidence_claims (
                        id, entity_type, entity_id, claim, source, source_url, source_excerpt, content_hash, confidence
                    ) VALUES (
                        gen_random_uuid(), 'user', :entity_id, 'Claim 2 duplicate', 'resume', '/path/resume.pdf', 'excerpt 2', :content_hash, 0.90
                    )
                    """
                ),
                {"entity_id": claim_entity_id, "content_hash": test_hash},
            )
        trans.rollback()

        # 5. Test automatic Brier score trigger computation
        trans = connection.begin()
        user_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        strategy_id = uuid.uuid4()
        action_id = uuid.uuid4()
        outcome_id = uuid.uuid4()

        connection.execute(
            text(
                """
                INSERT INTO users (id, email, full_name)
                VALUES (:user_id, 'candidate2@example.com', 'Taylor Candidate')
                """
            ),
            {"user_id": user_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO goals (id, user_id, objective_text, constraints_json, target_spec, success_criteria, sub_goals, deadline)
                VALUES (:goal_id, :user_id, 'Land Applied AI role', '{}', '{}', '{}', '[]', NOW() + INTERVAL '30 days')
                """
            ),
            {"goal_id": goal_id, "user_id": user_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO strategies (id, goal_id, version)
                VALUES (:strategy_id, :goal_id, 1)
                """
            ),
            {"strategy_id": strategy_id, "goal_id": goal_id},
        )
        # Action with predicted_probability = 0.8
        connection.execute(
            text(
                """
                INSERT INTO actions (
                    id, goal_id, strategy_id, action_type, target_type,
                    reason, predicted_outcome, predicted_probability, strategy_version
                ) VALUES (
                    :action_id, :goal_id, :strategy_id, 'reach_out', 'person',
                    'High response rate expected', 'Recruiter replies', 0.8, 'v1'
                )
                """
            ),
            {"action_id": action_id, "goal_id": goal_id, "strategy_id": strategy_id},
        )

        # Insert outcome with binary_success = False (0.0). Expected Brier: (0.8 - 0.0)^2 = 0.64
        # We pass dummy brier_score 0.0, trigger MUST overwrite it with (0.8 - 0.0)^2 = 0.64
        connection.execute(
            text(
                """
                INSERT INTO action_outcomes (
                    id, action_id, binary_success, brier_score
                ) VALUES (
                    :outcome_id, :action_id, false, 0.0
                )
                """
            ),
            {"outcome_id": outcome_id, "action_id": action_id},
        )

        row = connection.execute(
            text("SELECT brier_score FROM action_outcomes WHERE id = :outcome_id"),
            {"outcome_id": outcome_id},
        ).fetchone()

        assert row is not None
        assert pytest.approx(row[0], rel=1e-4) == 0.64
        trans.rollback()

    # 6. Test downgrade to base
    command.downgrade(alembic_cfg, "base")
