"""Unit tests for job board sourcing adapters, HTML stripping, rate limits, and retries."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from pilot.sourcing.ashby import AshbySource
from pilot.sourcing.base import (
    JobSourceNotFoundError,
    JobSourceRateLimitError,
)
from pilot.sourcing.greenhouse import GreenhouseSource
from pilot.sourcing.html_utils import strip_html_to_text
from pilot.sourcing.lever import LeverSource
from pilot.sourcing.schemas import SourceQuery


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


# ============================================================================
# HTML Stripper Tests
# ============================================================================
def test_strip_html_to_text():
    """Verify deterministic conversion of HTML, entities, and whitespace."""
    # Standard HTML
    html_input = "<h1>Title</h1><p>First paragraph with <strong>bold</strong> text.</p><ul><li>Item 1</li><li>Item 2</li></ul>"
    result = strip_html_to_text(html_input)
    assert "Title" in result
    assert "First paragraph with bold text." in result
    assert "- Item 1" in result
    assert "- Item 2" in result

    # Doubly escaped HTML
    escaped = "&lt;p&gt;Hello &amp;amp; Welcome to &lt;strong&gt;Pilot&lt;/strong&gt;&lt;/p&gt;"
    res_escaped = strip_html_to_text(escaped)
    assert "Hello & Welcome to Pilot" in res_escaped

    # None and empty
    assert strip_html_to_text(None) == ""
    assert strip_html_to_text("   ") == ""


# ============================================================================
# Greenhouse Adapter Tests
# ============================================================================
def test_greenhouse_adapter_normalization(tmp_path):
    """Verify Greenhouse adapter normalizes fixture payload with HTML, missing fields, and non-ASCII."""
    fixture_path = Path("tests/fixtures/sourcing/greenhouse.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixture_data = json.load(f)

    routes = {
        "GET:/v1/boards/canonical/jobs": (200, fixture_data, None),
    }
    transport, state = create_mock_transport(routes)
    source = GreenhouseSource(cache_dir=tmp_path, transport=transport)

    postings = source.fetch(SourceQuery(token="canonical", company_name="Canonical"))

    assert len(postings) == 3
    assert state["call_count"] == 1

    # Posting 1: Remote, HTML content stripped, raw preserved
    p1 = postings[0]
    assert p1.source == "greenhouse"
    assert p1.external_id == "101"
    assert p1.company_name == "Canonical"
    assert p1.title == "Senior Distributed Systems Engineer"
    assert p1.location == "Remote, US"
    assert p1.location_type == "remote"
    assert "5+ years Python and Go" in p1.description_text
    assert "<strong" not in p1.description_text
    assert p1.raw["id"] == 101
    assert p1.posting_url == "https://boards.greenhouse.io/canonical/jobs/101"

    # Posting 2: Non-ASCII text, European characters
    p2 = postings[1]
    assert p2.external_id == "102"
    assert "Ingénieur Données & IA 🚀 - Zurich" in p2.title
    assert p2.location == "Zürich, Switzerland"
    assert "PyTorch et Kafka" in p2.description_text

    # Posting 3: Missing location and missing description
    p3 = postings[2]
    assert p3.external_id == "103"
    assert p3.location is None
    assert p3.description_text is None
    assert p3.raw["id"] == 103


# ============================================================================
# Ashby Adapter Tests
# ============================================================================
def test_ashby_adapter_normalization(tmp_path):
    """Verify Ashby adapter normalizes fixture payload with remote flags, plain/HTML text, and non-ASCII."""
    fixture_path = Path("tests/fixtures/sourcing/ashby.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixture_data = json.load(f)

    routes = {
        "GET:/posting-api/job-board/ramp": (200, fixture_data, None),
    }
    transport, _ = create_mock_transport(routes)
    source = AshbySource(cache_dir=tmp_path, transport=transport)

    postings = source.fetch(SourceQuery(token="ramp", company_name="Ramp"))

    assert len(postings) == 3

    # Posting 1: isRemote True, descriptionPlain
    p1 = postings[0]
    assert p1.source == "ashby"
    assert p1.external_id == "ashby-201"
    assert p1.company_name == "Ramp"
    assert p1.location_type == "remote"
    assert "Build scalable financial infrastructure" in p1.description_text

    # Posting 2: Dict location with non-ASCII München, descriptionHtml stripped
    p2 = postings[1]
    assert p2.external_id == "ashby-202"
    assert "Lead Développeur Backend München" in p2.title
    assert p2.location == "München, Germany"
    assert "Leitung des Backend-Teams" in p2.description_text

    # Posting 3: Missing location and description
    p3 = postings[2]
    assert p3.external_id == "ashby-203"
    assert p3.location is None
    assert p3.description_text is None


# ============================================================================
# Lever Adapter Tests
# ============================================================================
def test_lever_adapter_normalization(tmp_path):
    """Verify Lever adapter normalizes fixture payload with workplace types, multi-part descriptions, and non-ASCII."""
    fixture_path = Path("tests/fixtures/sourcing/lever.json")
    with open(fixture_path, encoding="utf-8") as f:
        fixture_data = json.load(f)

    routes = {
        "GET:/v0/postings/palantir": (200, fixture_data, None),
    }
    transport, _ = create_mock_transport(routes)
    source = LeverSource(cache_dir=tmp_path, transport=transport)

    postings = source.fetch(SourceQuery(token="palantir", company_name="Palantir"))

    assert len(postings) == 3

    # Posting 1: remote workplaceType, combined description and additional text
    p1 = postings[0]
    assert p1.source == "lever"
    assert p1.external_id == "lever-301"
    assert p1.location_type == "remote"
    assert "Design and deploy machine learning pipelines" in p1.description_text
    assert "Benefits include health insurance" in p1.description_text

    # Posting 2: hybrid workplaceType, non-ASCII Portuguese text
    p2 = postings[1]
    assert p2.external_id == "lever-302"
    assert "Engenheiro de Dados Sênior — São Paulo" in p2.title
    assert p2.location == "São Paulo, Brazil"
    assert p2.location_type == "hybrid"
    assert "Desenvolver pipelines de dados" in p2.description_text

    # Posting 3: Missing categories and descriptions
    p3 = postings[2]
    assert p3.external_id == "lever-303"
    assert p3.location is None
    assert p3.description_text is None


# ============================================================================
# Resilience: Rate Limiting, Retries, and Errors
# ============================================================================
def test_adapter_rate_limit_header(tmp_path):
    """Verify JobSourceRateLimitError is raised when X-RateLimit-Remaining is 0."""
    routes = {
        "GET:/v1/boards/canonical/jobs": (
            200,
            {"jobs": []},
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"},
        ),
    }
    transport, _ = create_mock_transport(routes)
    source = GreenhouseSource(cache_dir=tmp_path, transport=transport)

    with pytest.raises(JobSourceRateLimitError, match="Rate limit exceeded"):
        source.fetch(SourceQuery(token="canonical"))


def test_adapter_rate_limit_429(tmp_path):
    """Verify JobSourceRateLimitError is raised on HTTP 429."""
    routes = {
        "GET:/posting-api/job-board/ramp": (429, {"message": "Too Many Requests"}, None),
    }
    transport, _ = create_mock_transport(routes)
    source = AshbySource(cache_dir=tmp_path, transport=transport)

    with pytest.raises(JobSourceRateLimitError, match="HTTP 429"):
        source.fetch(SourceQuery(token="ramp"))


def test_adapter_not_found_404(tmp_path):
    """Verify JobSourceNotFoundError is raised on HTTP 404."""
    routes = {
        "GET:/v0/postings/nonexistent": (404, {"message": "Not Found"}, None),
    }
    transport, _ = create_mock_transport(routes)
    source = LeverSource(cache_dir=tmp_path, transport=transport)

    with pytest.raises(JobSourceNotFoundError, match="Target resource not found"):
        source.fetch(SourceQuery(token="nonexistent"))


def test_adapter_5xx_retry_and_recovery(tmp_path):
    """Verify adapter retries on 500 server error and recovers on subsequent attempt."""
    call_count = 0

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(status_code=500, content=b"Internal Server Error")
        return httpx.Response(status_code=200, json={"jobs": []})

    transport = httpx.MockTransport(flaky_handler)
    source = GreenhouseSource(cache_dir=tmp_path, transport=transport)

    postings = source.fetch(SourceQuery(token="canonical"))
    assert postings == []
    assert call_count == 2


def test_adapter_5xx_exhaustion(tmp_path):
    """Verify adapter exhausts retries and raises error when server consistently returns 500."""
    call_count = 0

    def failing_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=503, content=b"Service Unavailable")

    transport = httpx.MockTransport(failing_handler)
    source = GreenhouseSource(cache_dir=tmp_path, transport=transport)

    with pytest.raises(httpx.HTTPStatusError):
        source.fetch(SourceQuery(token="canonical"))
    assert call_count == 4  # Initial attempt + 3 retries


def test_adapter_caching(tmp_path):
    """Verify cached responses prevent redundant HTTP network calls."""
    routes = {
        "GET:/v1/boards/canonical/jobs": (
            200,
            {"jobs": [{"id": 999, "title": "Cached Role"}]},
            None,
        ),
    }
    transport, state = create_mock_transport(routes)
    source = GreenhouseSource(cache_dir=tmp_path, transport=transport)

    # First fetch hits mock transport
    res1 = source.fetch(SourceQuery(token="canonical"))
    assert len(res1) == 1
    assert state["call_count"] == 1

    # Second fetch should hit cache on disk
    res2 = source.fetch(SourceQuery(token="canonical"))
    assert len(res2) == 1
    assert state["call_count"] == 1
