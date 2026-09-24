"""Alembic synchronous migration runner wired to Pilot Settings and Base.metadata."""

from logging.config import fileConfig

import pgvector.sqlalchemy
from alembic import context
from sqlalchemy import engine_from_config, pool

from pilot.config import get_settings
from pilot.db import models  # noqa: F401 - register all mapped models
from pilot.db.session import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Override sqlalchemy.url with secret value from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url.get_secret_value())


def compare_type(
    context,
    inspected_column,
    metadata_column,
    inspected_type,
    metadata_type,
):
    """Custom compare_type to avoid spurious diffs on pgvector Vector columns."""
    if isinstance(metadata_type, pgvector.sqlalchemy.Vector):
        if isinstance(inspected_type, pgvector.sqlalchemy.Vector):
            return metadata_type.dim != inspected_type.dim
        if getattr(inspected_type, "name", "").lower() == "vector":
            return False
    return None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=compare_type,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Register pgvector in dialect ischema_names so Alembic recognizes the type
        connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=compare_type,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
