"""Pytest fixtures for database testing."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pilot.config import get_settings


@pytest.fixture(scope="session")
def db_engine():
    """Create test engine from configuration."""
    settings = get_settings()
    engine = create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Provide a transactional session for a test function."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session_factory = sessionmaker(bind=connection, expire_on_commit=False)
    session = session_factory()

    yield session

    session.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()
