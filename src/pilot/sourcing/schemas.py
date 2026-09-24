"""Pydantic models for job sourcing queries, normalized postings, and upsert results."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceQuery(BaseModel):
    """Specification for fetching job postings from an ATS/board."""

    token: str
    company_name: str | None = None
    limit: int | None = None


class SourcedPosting(BaseModel):
    """Normalized job posting returned by a job board adapter."""

    source: str = Field(min_length=1, max_length=50)
    external_id: str = Field(min_length=1, max_length=255)
    company_name: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1, max_length=255)
    location: str | None = None
    location_type: str = Field(default="onsite", max_length=50)
    posting_url: str | None = None
    description_text: str | None = None
    posted_at: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SourcingResult(BaseModel):
    """Aggregate result from upserting sourced postings."""

    new_count: int = 0
    refreshed_count: int = 0
    closed_count: int = 0
    total_processed: int = 0
