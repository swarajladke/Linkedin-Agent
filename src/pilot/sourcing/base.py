"""Base protocol and exceptions for job board sourcing adapters."""

from typing import Protocol, runtime_checkable

from pilot.sourcing.schemas import SourcedPosting, SourceQuery


class JobSourceError(Exception):
    """Base exception for job board sourcing errors."""


class JobSourceRateLimitError(JobSourceError):
    """Raised when job board API rate limits are hit."""


class JobSourceNotFoundError(JobSourceError):
    """Raised when the specified board/org does not exist."""


@runtime_checkable
class JobSource(Protocol):
    """Protocol for ATS/job board sourcing adapters."""

    def fetch(self, query: SourceQuery) -> list[SourcedPosting]:
        """Fetch and normalize job postings for the given query."""
        ...
