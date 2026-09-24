"""Tests for public GitHub REST client, rate limit handling, retries, and source URLs."""

import json
from typing import Any

import httpx
import pytest

from pilot.ingestion.github import (
    GitHubClient,
    GitHubRateLimitError,
    GitHubUserNotFoundError,
)


def create_mock_transport(responses: dict[str, Any] | None = None, state: dict | None = None):
    """Create a mock transport returning predefined responses based on URL path."""
    routes = responses or {}
    shared_state = state if state is not None else {"call_count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        shared_state["call_count"] += 1
        path = request.url.path
        method = request.method

        key = f"{method}:{path}"
        if key in routes:
            route_data = routes[key]
            if callable(route_data):
                return route_data(request)
            status_code, content, headers = route_data
            return httpx.Response(
                status_code=status_code,
                content=json.dumps(content).encode()
                if isinstance(content, dict | list)
                else content.encode(),
                headers=headers or {"content-type": "application/json"},
                request=request,
            )

        return httpx.Response(status_code=404, json={"message": "Not Found"}, request=request)

    return httpx.MockTransport(handler), shared_state


def test_github_fetch_user_data_success(tmp_path):
    """Test fetching profile, repos, languages, README, and commit counts with proper source URLs."""
    routes = {
        "GET:/users/testdev": (
            200,
            {
                "login": "testdev",
                "name": "Test Developer",
                "bio": "AI systems engineer building agents.",
                "company": "AI Labs",
                "blog": "https://testdev.io",
                "location": "Bangalore",
                "public_repos": 3,
                "created_at": "2021-01-15T10:00:00Z",
                "html_url": "https://github.com/testdev",
            },
            {"X-RateLimit-Remaining": "50"},
        ),
        "GET:/users/testdev/repos": (
            200,
            [
                {
                    "name": "pilot-core",
                    "description": "Autonomous career agent",
                    "html_url": "https://github.com/testdev/pilot-core",
                    "language": "Python",
                    "languages_url": "https://api.github.com/repos/testdev/pilot-core/languages",
                    "stargazers_count": 12,
                    "forks_count": 2,
                    "topics": ["ai", "agents"],
                    "created_at": "2023-01-01T00:00:00Z",
                    "pushed_at": "2023-05-01T00:00:00Z",
                    "fork": False,
                    "archived": False,
                    "owner": {"login": "testdev"},
                },
                {
                    "name": "forked-repo",
                    "description": "A forked project",
                    "html_url": "https://github.com/testdev/forked-repo",
                    "language": "Go",
                    "fork": True,
                    "archived": False,
                    "created_at": "2022-01-01T00:00:00Z",
                    "owner": {"login": "testdev"},
                },
                {
                    "name": "archived-repo",
                    "description": "Old deprecated project",
                    "html_url": "https://github.com/testdev/archived-repo",
                    "language": "Python",
                    "fork": False,
                    "archived": True,
                    "created_at": "2020-01-01T00:00:00Z",
                    "owner": {"login": "testdev"},
                },
            ],
            {"X-RateLimit-Remaining": "49"},
        ),
        "GET:/repos/testdev/pilot-core/languages": (
            200,
            {"Python": 95000, "Shell": 5000},
            {"X-RateLimit-Remaining": "48"},
        ),
        "GET:/repos/testdev/pilot-core/readme": (
            200,
            "# Pilot Core\nAutonomous career agent loop.",
            {"X-RateLimit-Remaining": "47", "content-type": "text/plain"},
        ),
        "GET:/repos/testdev/pilot-core/commits": (
            200,
            [{"sha": "abc1234"}],
            {
                "X-RateLimit-Remaining": "46",
                "Link": '<https://api.github.com/repos/testdev/pilot-core/commits?author=testdev&per_page=1&page=38>; rel="last"',
            },
        ),
    }

    transport, state = create_mock_transport(routes)
    client = GitHubClient(cache_dir=tmp_path / ".cache", transport=transport)

    data = client.fetch_user_data("testdev")

    # Profile assertions
    assert data.username == "testdev"
    assert data.profile.html_url == "https://github.com/testdev"
    assert data.source_url == "https://github.com/testdev"
    assert data.profile.name == "Test Developer"

    # Default filtering: fork and archived repos excluded
    assert len(data.repos) == 1
    repo = data.repos[0]
    assert repo.name == "pilot-core"
    assert repo.html_url == "https://github.com/testdev/pilot-core"
    assert repo.languages == {"Python": 95000, "Shell": 5000}
    assert repo.readme == "# Pilot Core\nAutonomous career agent loop."
    assert repo.readme_url == "https://github.com/testdev/pilot-core#readme"
    assert repo.user_commit_count == 38

    # Second call should be served from cache without extra HTTP calls
    call_count_before = state["call_count"]
    cached_data = client.fetch_user_data("testdev")
    assert state["call_count"] == call_count_before
    assert len(cached_data.repos) == 1


def test_github_include_forks_and_archived(tmp_path):
    """Test including forks and archived repos when configured."""
    routes = {
        "GET:/users/testdev": (
            200,
            {
                "login": "testdev",
                "public_repos": 2,
                "created_at": "2021-01-15T10:00:00Z",
                "html_url": "https://github.com/testdev",
            },
            {},
        ),
        "GET:/users/testdev/repos": (
            200,
            [
                {
                    "name": "forked-repo",
                    "html_url": "https://github.com/testdev/forked-repo",
                    "fork": True,
                    "archived": False,
                    "created_at": "2022-01-01T00:00:00Z",
                    "owner": {"login": "testdev"},
                },
                {
                    "name": "archived-repo",
                    "html_url": "https://github.com/testdev/archived-repo",
                    "fork": False,
                    "archived": True,
                    "created_at": "2020-01-01T00:00:00Z",
                    "owner": {"login": "testdev"},
                },
            ],
            {},
        ),
        "GET:/repos/testdev/archived-repo/readme": (404, {"message": "Not Found"}, {}),
        "GET:/repos/testdev/archived-repo/commits": (200, [], {}),
    }

    transport, _ = create_mock_transport(routes)
    client = GitHubClient(cache_dir=tmp_path / ".cache", transport=transport)

    data = client.fetch_user_data("testdev", exclude_forks=False, exclude_archived=False)
    assert len(data.repos) == 2


def test_github_rate_limit_error(tmp_path):
    """Test that exhausted rate limit raises GitHubRateLimitError."""
    routes = {
        "GET:/users/testdev": (
            403,
            {"message": "API rate limit exceeded for user"},
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"},
        ),
    }

    transport, _ = create_mock_transport(routes)
    client = GitHubClient(cache_dir=tmp_path / ".cache", transport=transport)

    with pytest.raises(GitHubRateLimitError) as exc_info:
        client.fetch_user_data("testdev")

    assert "rate limit exceeded" in str(exc_info.value).lower()


def test_github_retry_on_5xx_server_error(tmp_path):
    """Test exponential backoff retries when encountering 5xx errors."""
    attempts = {"count": 0}

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(status_code=502, content=b"Bad Gateway", request=request)
        return httpx.Response(
            status_code=200,
            json={
                "login": "testdev",
                "public_repos": 0,
                "created_at": "2021-01-15T10:00:00Z",
                "html_url": "https://github.com/testdev",
            },
            request=request,
        )

    routes = {
        "GET:/users/testdev": flaky_handler,
        "GET:/users/testdev/repos": (200, [], {}),
    }

    transport, _ = create_mock_transport(routes)
    client = GitHubClient(cache_dir=tmp_path / ".cache", transport=transport)

    data = client.fetch_user_data("testdev")
    assert data.username == "testdev"
    assert attempts["count"] == 3


def test_github_user_not_found(tmp_path):
    """Test that nonexistent user raises GitHubUserNotFoundError."""
    routes = {
        "GET:/users/nonexistent": (404, {"message": "Not Found"}, {}),
    }

    transport, _ = create_mock_transport(routes)
    client = GitHubClient(cache_dir=tmp_path / ".cache", transport=transport)

    with pytest.raises(GitHubUserNotFoundError):
        client.fetch_user_data("nonexistent")
