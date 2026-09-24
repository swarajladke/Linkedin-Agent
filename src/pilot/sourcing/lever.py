"""Lever job board sourcing adapter."""

from datetime import UTC, datetime
from typing import Any

from pilot.sourcing.html_utils import strip_html_to_text
from pilot.sourcing.http_adapter import BaseHttpAdapter
from pilot.sourcing.schemas import SourcedPosting, SourceQuery


class LeverSource(BaseHttpAdapter):
    """Adapter for public Lever Job Board API."""

    BASE_URL = "https://api.lever.co/v0/postings"

    def fetch(self, query: SourceQuery, force_refresh: bool = False) -> list[SourcedPosting]:
        """Fetch and normalize job postings from Lever."""
        url = f"{self.BASE_URL}/{query.token}?mode=json"
        data = self._request_json(
            url=url, namespace="lever", cache_key=query.token, force_refresh=force_refresh
        )

        jobs_list: list[Any] = []
        if isinstance(data, list):
            jobs_list = data
        elif isinstance(data, dict):
            jobs_list = data.get("postings", data.get("data", []))

        postings: list[SourcedPosting] = []

        for item in jobs_list:
            if not isinstance(item, dict):
                continue

            external_id = str(item.get("id", ""))
            if not external_id:
                continue

            title = str(item.get("text") or "Untitled Role").strip()

            categories = item.get("categories") or {}
            location_str: str | None = None
            workplace_type = ""
            if isinstance(categories, dict):
                location_str = categories.get("location")
                workplace_type = str(categories.get("workplaceType", "")).lower()

            loc_lower = (location_str or "").lower()
            title_lower = title.lower()

            if workplace_type == "remote" or "remote" in loc_lower or "remote" in title_lower:
                location_type = "remote"
            elif "hybrid" in workplace_type or "hybrid" in loc_lower:
                location_type = "hybrid"
            else:
                location_type = "onsite"

            desc_parts: list[str] = []
            desc_plain = item.get("descriptionPlain")
            desc_raw = item.get("description")
            if desc_plain and str(desc_plain).strip():
                desc_parts.append(str(desc_plain).strip())
            elif desc_raw:
                desc_parts.append(strip_html_to_text(desc_raw))

            additional_plain = item.get("additionalPlain")
            additional_raw = item.get("additional")
            if additional_plain and str(additional_plain).strip():
                desc_parts.append(str(additional_plain).strip())
            elif additional_raw:
                desc_parts.append(strip_html_to_text(additional_raw))

            description_text = "\n\n".join(desc_parts) if desc_parts else None

            posted_at: datetime | None = None
            created_at = item.get("createdAt")
            if isinstance(created_at, int | float):
                try:
                    posted_at = datetime.fromtimestamp(created_at / 1000.0, tz=UTC)
                except (ValueError, OSError):
                    posted_at = datetime.now(UTC)
            elif isinstance(created_at, str):
                try:
                    posted_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    posted_at = datetime.now(UTC)

            company_name = query.company_name or query.token.capitalize()
            posting_url = item.get("hostedUrl") or item.get("applyUrl")

            postings.append(
                SourcedPosting(
                    source="lever",
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
