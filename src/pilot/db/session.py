"""Synchronous database session and engine setup with explicit metadata naming conventions."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pilot.config import get_settings

# Stable naming conventions for Alembic autogenerate
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base declarative class with explicit naming convention metadata."""

    metadata = metadata


def get_engine():
    """Create a synchronous SQLAlchemy engine using settings."""
    settings = get_settings()
    return create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        echo=False,
    )


def get_session_factory():
    """Create a scoped sessionmaker bound to the engine."""
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional synchronous database session scope."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
