"""Ingestion module for Pilot: document parsing and public data source fetching."""

from pilot.ingestion.github import (
    GitHubClient,
    GitHubClientError,
    GitHubProfile,
    GitHubRateLimitError,
    GitHubRepo,
    GitHubUserData,
    GitHubUserNotFoundError,
)
from pilot.ingestion.reader import (
    EncryptedPDFError,
    ImageOnlyPDFError,
    ResumeDocument,
    ResumeReader,
    ResumeReaderError,
    SourceSpan,
    UnsupportedFileFormatError,
)

__all__ = [
    "SourceSpan",
    "ResumeDocument",
    "ResumeReader",
    "ResumeReaderError",
    "EncryptedPDFError",
    "ImageOnlyPDFError",
    "UnsupportedFileFormatError",
    "GitHubClient",
    "GitHubClientError",
    "GitHubProfile",
    "GitHubRepo",
    "GitHubUserData",
    "GitHubRateLimitError",
    "GitHubUserNotFoundError",
]
