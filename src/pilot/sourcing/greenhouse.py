"""Greenhouse job board sourcing adapter."""

from datetime import UTC, datetime

from pilot.sourcing.html_utils import strip_html_to_text
from pilot.sourcing.http_adapter import BaseHttpAdapter
from pilot.sourcing.schemas import SourcedPosting, SourceQuery


class GreenhouseSource(BaseHttpAdapter):
    """Adapter for public Greenhouse Job Board API."""

    BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

    def fetch(self, query: SourceQuery, force_refresh: bool = False) -> list[SourcedPosting]:
        """Fetch and normalize job postings from Greenhouse."""
        url = f"{self.BASE_URL}/{query.token}/jobs?content=true"
        data = self._request_json(
            url=url, namespace="greenhouse", cache_key=query.token, force_refresh=force_refresh
        )

        jobs_list = data.get("jobs", []) if isinstance(data, dict) else []
        postings: list[SourcedPosting] = []

        for item in jobs_list:
            if not isinstance(item, dict):
                continue

            external_id = str(item.get("id", ""))
            if not external_id:
                continue

            title = str(item.get("title") or "Untitled Role").strip()
            raw_loc = item.get("location")
            location_str: str | None = None
            if isinstance(raw_loc, dict):
                location_str = raw_loc.get("name")
            elif isinstance(raw_loc, str):
                location_str = raw_loc

            location_type = "onsite"
            loc_lower = (location_str or "").lower()
            title_lower = title.lower()
            if "remote" in loc_lower or "remote" in title_lower:
                location_type = "remote"
            elif "hybrid" in loc_lower:
                location_type = "hybrid"

            raw_content = item.get("content")
            description_text = strip_html_to_text(raw_content) if raw_content else None

            posted_at: datetime | None = None
            updated_at_str = item.get("updated_at")
            if updated_at_str:
                try:
                    posted_at = datetime.fromisoformat(str(updated_at_str).replace("Z", "+00:00"))
                except ValueError:
                    posted_at = datetime.now(UTC)

            company_name = query.company_name or query.token.capitalize()
            posting_url = item.get("absolute_url")

            postings.append(
                SourcedPosting(
                    source="greenhouse",
                    external_id=external_id,
                    company_name=company_name,
                    title=title,
                    location=location_str,
                    location_type=location_type,
                    posting_url=posting_url,
                    description_text=description_text,
                    posted_at=posted_at,
                    raw=item,
                )
            )

            if query.limit and len(postings) >= query.limit:
                break

        return postings
