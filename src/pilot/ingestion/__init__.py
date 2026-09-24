"""Ingestion module for Pilot: document parsing and public data source fetching."""

from pilot.ingestion.github import (
    GitHubClient,
    GitHubProfile,
    GitHubRateLimitError,
    GitHubRepo,
    GitHubUserData,
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
    "GitHubProfile",
    "GitHubRepo",
    "GitHubUserData",
    "GitHubRateLimitError",
]
