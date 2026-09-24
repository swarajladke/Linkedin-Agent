"""Job intelligence sourcing module."""

from pilot.sourcing.ashby import AshbySource
from pilot.sourcing.base import (
    JobSource,
    JobSourceError,
    JobSourceNotFoundError,
    JobSourceRateLimitError,
)
from pilot.sourcing.greenhouse import GreenhouseSource
from pilot.sourcing.lever import LeverSource
from pilot.sourcing.repository import get_or_create_company, upsert_roles
from pilot.sourcing.schemas import SourcedPosting, SourceQuery, SourcingResult

__all__ = [
    "AshbySource",
    "GreenhouseSource",
    "JobSource",
    "JobSourceError",
    "JobSourceNotFoundError",
    "JobSourceRateLimitError",
    "LeverSource",
    "SourceQuery",
    "SourcedPosting",
    "SourcingResult",
    "get_or_create_company",
    "upsert_roles",
]
