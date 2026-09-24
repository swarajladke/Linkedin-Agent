"""Integration tests for job sourcing idempotency, timestamp progression, and role closure."""

import json
import time
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Role, RoleStatus
from pilot.sourcing.greenhouse import GreenhouseSource
from pilot.sourcing.repository import upsert_roles
from pilot.sourcing.schemas import SourceQuery

pytestmark = pytest.mark.integration


def test_sourcing_idempotency_timestamp_progression_and_role_closure(session: Session, tmp_path):
    """
    Integration Invariant:
    - Sourcing the same fixture board three times yields stable row counts,
      stable first_seen_at, and advancing last_seen_at.
    - A role that vanishes from the feed transitions to CLOSED and is not deleted.
    """
    fixture_path = Path("tests/fixtures/sourcing/greenhouse.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixture_data = json.load(f)

    # 1. First sourcing run: 3 roles discovered
    from tests.test_sourcing import create_mock_transport

    routes = {"GET:/v1/boards/canonical/jobs": (200, fixture_data, None)}
    transport, _ = create_mock_transport(routes)
    source = GreenhouseSource(cache_dir=tmp_path / "cache1", transport=transport)
    postings_run1 = source.fetch(SourceQuery(token="canonical", company_name="Canonical"))

    res1 = upsert_roles(session, postings_run1)
    session.commit()

    assert res1.new_count == 3
    assert res1.refreshed_count == 0
    assert res1.closed_count == 0

    roles_run1 = session.scalars(select(Role).order_by(Role.external_id)).all()
    assert len(roles_run1) == 3
    initial_first_seen = {r.external_id: r.first_seen_at for r in roles_run1}
    initial_last_seen = {r.external_id: r.last_seen_at for r in roles_run1}

    # Ensure slight delay so timestamps advance
    time.sleep(0.05)

    # 2. Second sourcing run with identical feed: Idempotent refresh
    source2 = GreenhouseSource(cache_dir=tmp_path / "cache2", transport=transport)
    postings_run2 = source2.fetch(SourceQuery(token="canonical", company_name="Canonical"))

    res2 = upsert_roles(session, postings_run2)
    session.commit()

    assert res2.new_count == 0
    assert res2.refreshed_count == 3
    assert res2.closed_count == 0

    roles_run2 = session.scalars(select(Role).order_by(Role.external_id)).all()
    assert len(roles_run2) == 3  # Stable row count

    for r in roles_run2:
        # Stable first_seen_at
        assert r.first_seen_at == initial_first_seen[r.external_id]
        # Advancing last_seen_at
        assert r.last_seen_at > initial_last_seen[r.external_id]
        assert r.status == RoleStatus.OPEN

    second_last_seen = {r.external_id: r.last_seen_at for r in roles_run2}
    time.sleep(0.05)

    # 3. Third sourcing run: Same board again, but role 103 vanishes from feed
    postings_run3 = [p for p in postings_run2 if p.external_id != "103"]
    assert len(postings_run3) == 2

    res3 = upsert_roles(session, postings_run3)
    session.commit()

    assert res3.new_count == 0
    assert res3.refreshed_count == 2
    assert res3.closed_count == 1  # Role 103 transitioned to CLOSED

    roles_run3 = session.scalars(select(Role).order_by(Role.external_id)).all()
    assert len(roles_run3) == 3  # Role 103 is NOT deleted, row count remains 3!

    role_101 = next(r for r in roles_run3 if r.external_id == "101")
    role_102 = next(r for r in roles_run3 if r.external_id == "102")
    role_103 = next(r for r in roles_run3 if r.external_id == "103")

    assert role_101.status == RoleStatus.OPEN
    assert role_101.first_seen_at == initial_first_seen["101"]
    assert role_101.last_seen_at > second_last_seen["101"]

    assert role_102.status == RoleStatus.OPEN
    assert role_102.first_seen_at == initial_first_seen["102"]
    assert role_102.last_seen_at > second_last_seen["102"]

    # Role 103 vanished and is marked CLOSED
    assert role_103.status == RoleStatus.CLOSED
    assert role_103.first_seen_at == initial_first_seen["103"]

    # 4. Fourth sourcing run: Role 103 reappears in feed
    res4 = upsert_roles(session, postings_run2)
    session.commit()

    assert res4.new_count == 0
    assert res4.refreshed_count == 3
    assert res4.closed_count == 0

    reopened_103 = session.scalars(select(Role).where(Role.external_id == "103")).one()
    assert reopened_103.status == RoleStatus.OPEN
