"""Public GitHub REST client for candidate portfolio and code ingestion with strict provenance."""

import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from pilot.config import get_settings
from pilot.ingestion.cache import IngestionCache


# ============================================================================
# Models & Errors
# ============================================================================
class GitHubProfile(BaseModel):
    """Candidate GitHub user profile."""

    username: str
    name: str | None = None
    bio: str | None = None
    company: str | None = None
    blog: str | None = None
    location: str | None = None
    public_repos: int = 0
    created_at: datetime
    html_url: str


class GitHubRepo(BaseModel):
    """Repository metadata, language breakdown, README, and user contributions."""

    name: str
    description: str | None = None
    html_url: str
    language: str | None = None
    languages: dict[str, int] = Field(default_factory=dict)
    stargazers_count: int = 0
    forks_count: int = 0
    topics: list[str] = Field(default_factory=list)
    created_at: datetime
    pushed_at: datetime | None = None
    fork: bool = False
    archived: bool = False
    readme: str | None = None
    readme_url: str | None = None
    user_commit_count: int | None = None


class GitHubUserData(BaseModel):
    """Aggregated portfolio and code data for a candidate."""

    username: str
    profile: GitHubProfile
    repos: list[GitHubRepo]
    fetched_at: datetime
    source_url: str


class GitHubClientError(Exception):
    """Base exception for GitHub client errors."""


class GitHubUserNotFoundError(GitHubClientError):
    """Raised when GitHub user does not exist."""


class GitHubRateLimitError(GitHubClientError):
    """Raised when GitHub API rate limits are exhausted."""


# ============================================================================
# GitHub Client
# ============================================================================
class GitHubClient:
    """Public GitHub REST client with rate limit awareness, retries, and caching."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        cache_dir: str | Path = ".cache",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        settings = get_settings()
        self.token = token or (
            settings.github_token.get_secret_value() if settings.github_token else None
        )
        self.cache = IngestionCache(cache_dir=cache_dir)
        self.timeout = httpx.Timeout(timeout, connect=5.0)
        self.transport = transport

    def _get_headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Pilot-Autonomous-Career-Agent",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
        max_retries: int = 3,
    ) -> httpx.Response:
        """Execute request with rate limit checks and exponential backoff on 5xx."""
        headers = self._get_headers(accept=accept)
        backoff = 0.2

        for attempt in range(max_retries + 1):
            try:
                response = client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Rate Limit Inspection
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining is not None and int(remaining) == 0:
                    reset_ts = response.headers.get("X-RateLimit-Reset", "0")
                    raise GitHubRateLimitError(
                        f"GitHub rate limit exceeded. Reset timestamp: {reset_ts}."
                    )

                if response.status_code == 403 and "rate limit" in response.text.lower():
                    raise GitHubRateLimitError(f"GitHub rate limit exceeded: {response.text}")

                # Retry on 5xx server errors
                if response.status_code >= 500:
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    response.raise_for_status()

                return response

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise GitHubClientError(f"Network error communicating with GitHub: {exc}") from exc

        raise GitHubClientError(f"Failed to fetch {url} after {max_retries} retries")

    def fetch_user_data(
        self,
        username: str,
        max_repos: int = 50,
        exclude_forks: bool = True,
        exclude_archived: bool = True,
        force_refresh: bool = False,
    ) -> GitHubUserData:
        """Fetch candidate profile, public repos, language breakdown, READMEs, and commit counts."""
        cache_key = f"user:{username}:max_{max_repos}:forks_{exclude_forks}:arch_{exclude_archived}"
        if not force_refresh:
            cached = self.cache.get("github", cache_key)
            if cached:
                return GitHubUserData.model_validate(cached)

        with httpx.Client(transport=self.transport) as client:
            # 1. Fetch Profile
            profile_url = f"{self.BASE_URL}/users/{username}"
            resp = self._request(client, "GET", profile_url)
            if resp.status_code == 404:
                raise GitHubUserNotFoundError(f"GitHub user '{username}' was not found.")
            resp.raise_for_status()
            pdata = resp.json()

            profile = GitHubProfile(
                username=pdata["login"],
                name=pdata.get("name"),
                bio=pdata.get("bio"),
                company=pdata.get("company"),
                blog=pdata.get("blog") or None,
                location=pdata.get("location"),
                public_repos=pdata.get("public_repos", 0),
                created_at=datetime.fromisoformat(pdata["created_at"].replace("Z", "+00:00")),
                html_url=pdata["html_url"],
            )

            # 2. Fetch Repositories
            repos_url = f"{self.BASE_URL}/users/{username}/repos"
            params = {
                "sort": "pushed",
                "direction": "desc",
                "per_page": min(max_repos, 100),
            }
            resp = self._request(client, "GET", repos_url, params=params)
            resp.raise_for_status()
            raw_repos = resp.json()

            parsed_repos: list[GitHubRepo] = []
            for r in raw_repos:
                is_fork = r.get("fork", False)
                is_archived = r.get("archived", False)

                if exclude_forks and is_fork:
                    continue
                if exclude_archived and is_archived:
                    continue

                repo_name = r["name"]
                owner = r["owner"]["login"]
                repo_html_url = r["html_url"]

                # Fetch Language Breakdown
                languages: dict[str, int] = {}
                lang_url = r.get("languages_url")
                if lang_url:
                    lresp = self._request(client, "GET", lang_url)
                    if lresp.status_code == 200:
                        languages = lresp.json()

                # For non-fork repos: fetch README & user commits
                readme_text: str | None = None
                readme_url: str | None = None
                user_commits: int | None = None

                if not is_fork:
                    # README
                    readme_api_url = f"{self.BASE_URL}/repos/{owner}/{repo_name}/readme"
                    rm_resp = self._request(
                        client,
                        "GET",
                        readme_api_url,
                        accept="application/vnd.github.raw+json",
                    )
                    if rm_resp.status_code == 200:
                        readme_text = rm_resp.text
                        readme_url = f"{repo_html_url}#readme"

                    # Commits by author
                    commits_api_url = f"{self.BASE_URL}/repos/{owner}/{repo_name}/commits"
                    c_resp = self._request(
                        client,
                        "GET",
                        commits_api_url,
                        params={"author": username, "per_page": 1},
                    )
                    if c_resp.status_code == 200:
                        user_commits = self._extract_commit_count(c_resp)

                pushed_at_str = r.get("pushed_at")
                pushed_at = (
                    datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
                    if pushed_at_str
                    else None
                )

                repo_obj = GitHubRepo(
                    name=repo_name,
                    description=r.get("description"),
                    html_url=repo_html_url,
                    language=r.get("language"),
                    languages=languages,
                    stargazers_count=r.get("stargazers_count", 0),
                    forks_count=r.get("forks_count", 0),
                    topics=r.get("topics", []),
                    created_at=datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                    pushed_at=pushed_at,
                    fork=is_fork,
                    archived=is_archived,
                    readme=readme_text,
                    readme_url=readme_url,
                    user_commit_count=user_commits,
                )
                parsed_repos.append(repo_obj)
                if len(parsed_repos) >= max_repos:
                    break

            user_data = GitHubUserData(
                username=username,
                profile=profile,
                repos=parsed_repos,
                fetched_at=datetime.now(UTC),
                source_url=profile.html_url,
            )

            # Cache the aggregate response
            self.cache.set("github", cache_key, user_data.model_dump(mode="json"))
            return user_data

    def _extract_commit_count(self, response: httpx.Response) -> int:
        """Extract total commit count from Link header or length of response list."""
        link_header = response.headers.get("Link")
        if link_header:
            match = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link_header)
            if match:
                return int(match.group(1))

        data = response.json()
        return len(data) if isinstance(data, list) else 0
