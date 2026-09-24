"""Tests for github_to_spans asserting real URLs, locators, and README groundability."""

from datetime import UTC, datetime

from pilot.extraction.grounding import GroundingValidator
from pilot.extraction.spans import github_to_spans
from pilot.ingestion.github import GitHubProfile, GitHubRepo, GitHubUserData


def test_github_spans_generation_and_groundability():
    """Test that github_to_spans generates spans with proper URLs, locators, and groundable text."""
    now = datetime.now(UTC)
    profile = GitHubProfile(
        username="alexcandidate",
        name="Alex Candidate",
        bio="AI Engineer passionate about LLM inference and agentic workflows.",
        company="AI Research Lab",
        location="Bangalore",
        public_repos=1,
        created_at=now,
        html_url="https://github.com/alexcandidate",
    )
    repo = GitHubRepo(
        name="pilot-agent",
        description="Goal-directed autonomous career agent in Python.",
        html_url="https://github.com/alexcandidate/pilot-agent",
        language="Python",
        languages={"Python": 80000, "SQL": 15000},
        topics=["agent", "ai", "career"],
        created_at=now,
        pushed_at=now,
        fork=False,
        archived=False,
        readme="# Pilot Agent\n\nBuilt with Python 3.11 and PostgreSQL with pgvector.",
        readme_url="https://github.com/alexcandidate/pilot-agent#readme",
        user_commit_count=42,
    )
    data = GitHubUserData(
        username="alexcandidate",
        profile=profile,
        repos=[repo],
        fetched_at=now,
        source_url=profile.html_url,
    )

    spans = github_to_spans(data)
    assert len(spans) >= 6

    # Verify URLs
    for span in spans:
        assert span.source_url.startswith("https://github.com/alexcandidate")
        assert len(span.locator) > 0

    # Verify locators
    locators = [s.locator for s in spans]
    assert "profile;field=name" in locators
    assert "profile;field=bio" in locators
    assert "repo=pilot-agent;field=description" in locators
    assert "repo=pilot-agent;field=languages" in locators
    assert "repo=pilot-agent;field=commits" in locators
    assert any("file=README" in loc for loc in locators)

    # Test README text is groundable by GroundingValidator
    validator = GroundingValidator(spans)
    assert validator.is_grounded("PostgreSQL with pgvector") is True
    assert validator.is_grounded("42 commits to repository pilot-agent") is True
    assert validator.is_grounded("Goal-directed autonomous career agent") is True
