"""Deterministic HTML-to-text stripper for job descriptions."""

import html
import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._pieces.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
            self._pieces.append("\n\n")
        elif tag_lower in ("br", "li"):
            self._pieces.append("\n- " if tag_lower == "li" else "\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pieces.append("\n")

    def get_text(self) -> str:
        raw_text = "".join(self._pieces)
        # Unescape HTML entities
        unescaped = html.unescape(raw_text)
        # Normalize multiple spaces per line, preserve newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in unescaped.splitlines()]
        # Collapse multiple empty lines into at most double newlines
        collapsed: list[str] = []
        for line in lines:
            if line:
                collapsed.append(line)
            elif collapsed and collapsed[-1] != "":
                collapsed.append("")
        return "\n".join(collapsed).strip()


def strip_html_to_text(html_content: str | None) -> str:
    """Deterministically convert HTML content to clean, readable plain text."""
    if not html_content or not html_content.strip():
        return ""

    # If doubly escaped (e.g. &lt;p&gt;...), unescape first pass
    content = html_content
    if "&lt;" in content or "&gt;" in content:
        content = html.unescape(content)

    parser = _TextExtractor()
    parser.feed(content)
    parser.close()
    return parser.get_text()
