"""Ashby job board sourcing adapter."""

from datetime import UTC, datetime

from pilot.sourcing.html_utils import strip_html_to_text
from pilot.sourcing.http_adapter import BaseHttpAdapter
from pilot.sourcing.schemas import SourcedPosting, SourceQuery


class AshbySource(BaseHttpAdapter):
    """Adapter for public Ashby Job Board API."""

    BASE_URL = "https://api.ashbyhq.com/posting-api/job-board"

    def fetch(self, query: SourceQuery, force_refresh: bool = False) -> list[SourcedPosting]:
        """Fetch and normalize job postings from Ashby."""
        url = f"{self.BASE_URL}/{query.token}"
        data = self._request_json(
            url=url, namespace="ashby", cache_key=query.token, force_refresh=force_refresh
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
                location_str = raw_loc.get("name") or raw_loc.get("locationName")
            elif isinstance(raw_loc, str):
                location_str = raw_loc

            is_remote = bool(item.get("isRemote", False))
            loc_lower = (location_str or "").lower()
            title_lower = title.lower()

            if is_remote or "remote" in loc_lower or "remote" in title_lower:
                location_type = "remote"
            elif "hybrid" in loc_lower:
                location_type = "hybrid"
            else:
                location_type = "onsite"

            desc_plain = item.get("descriptionPlain")
            desc_html = item.get("descriptionHtml")
            description_text: str | None = None
            if desc_plain and str(desc_plain).strip():
                description_text = str(desc_plain).strip()
            elif desc_html:
                description_text = strip_html_to_text(desc_html)

            posted_at: datetime | None = None
            published_at = item.get("publishedAt")
            if published_at:
                try:
                    posted_at = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                except ValueError:
                    posted_at = datetime.now(UTC)

            company_name = query.company_name or query.token.capitalize()
            posting_url = item.get("jobUrl") or item.get("applyUrl")

            postings.append(
                SourcedPosting(
                    source="ashby",
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
