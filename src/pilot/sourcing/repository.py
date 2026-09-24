"""Database operations for sourcing: company deduplication and idempotent role upserts."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.db.models import Company, Role, RoleStatus
from pilot.sourcing.schemas import SourcedPosting, SourcingResult


def get_or_create_company(
    session: Session,
    name: str,
    domain: str | None = None,
    careers_url: str | None = None,
) -> Company:
    """Find company by name (case-insensitive) or create a new one."""
    clean_name = name.strip()
    stmt = select(Company).where(Company.name.ilike(clean_name))
    company = session.execute(stmt).scalars().first()

    if not company and domain:
        stmt_domain = select(Company).where(Company.domain == domain.strip().lower())
        company = session.execute(stmt_domain).scalars().first()

    if not company:
        company = Company(
            name=clean_name,
            domain=domain.strip().lower() if domain else None,
            careers_url=careers_url,
        )
        session.add(company)
        session.flush()

    return company


def upsert_roles(
    session: Session,
    postings: list[SourcedPosting],
    *,
    source: str | None = None,
    company_id: UUID | None = None,
) -> SourcingResult:
    """
    Upsert companies and roles with provenance, idempotently tracking active/closed status.

    - Upserts company by name.
    - Upserts role on (source, external_id).
    - Preserves first_seen_at, updates last_seen_at.
    - Marks roles absent from the fetched postings for the touched (source, company) as CLOSED.
    """
    now = datetime.now(UTC)
    result = SourcingResult(total_processed=len(postings))

    # Track seen external_ids grouped by (source, company_id)
    seen_by_source_company: dict[tuple[str, UUID], set[str]] = {}
    touched_companies: dict[str, Company] = {}

    for posting in postings:
        # 1. Resolve company
        comp_name = posting.company_name.strip()
        if comp_name not in touched_companies:
            company = get_or_create_company(
                session=session,
                name=comp_name,
                careers_url=posting.posting_url,
            )
            touched_companies[comp_name] = company
        else:
            company = touched_companies[comp_name]

        group_key = (posting.source, company.id)
        if group_key not in seen_by_source_company:
            seen_by_source_company[group_key] = set()
        seen_by_source_company[group_key].add(posting.external_id)

        # 2. Check for existing role
        stmt = select(Role).where(
            Role.source == posting.source,
            Role.external_id == posting.external_id,
        )
        existing_role = session.execute(stmt).scalars().first()

        if existing_role:
            # Refresh existing role
            existing_role.last_seen_at = now
            existing_role.title = posting.title
            existing_role.location = posting.location
            existing_role.location_type = posting.location_type
            existing_role.posting_url = posting.posting_url
            existing_role.requirements_summary = posting.description_text
            existing_role.raw_posting = posting.raw

            # Reopen if previously closed
            if existing_role.status == RoleStatus.CLOSED:
                existing_role.status = RoleStatus.OPEN

            result.refreshed_count += 1
        else:
            # Create new role
            new_role = Role(
                company_id=company.id,
                title=posting.title,
                location_type=posting.location_type,
                location=posting.location,
                posting_url=posting.posting_url,
                requirements_summary=posting.description_text,
                status=RoleStatus.OPEN,
                source=posting.source,
                external_id=posting.external_id,
                raw_posting=posting.raw,
                first_seen_at=now,
                last_seen_at=now,
            )
            session.add(new_role)
            result.new_count += 1

    session.flush()

    # 3. Mark absent roles as CLOSED
    # If source and company_id explicitly provided, ensure group is tracked even if postings is empty
    if source and company_id:
        group_key = (source, company_id)
        if group_key not in seen_by_source_company:
            seen_by_source_company[group_key] = set()

    for (src, comp_id), present_ids in seen_by_source_company.items():
        stmt_open = select(Role).where(
            Role.source == src,
            Role.company_id == comp_id,
            Role.status != RoleStatus.CLOSED,
        )
        open_roles = session.execute(stmt_open).scalars().all()
        for role in open_roles:
            if role.external_id not in present_ids:
                role.status = RoleStatus.CLOSED
                result.closed_count += 1

    session.flush()
    return result
