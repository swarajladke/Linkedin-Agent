"""Strict grounding validator verifying that extracted claims verbatim-match source document spans."""

from dataclasses import dataclass

from pilot.ingestion.reader import SourceSpan


@dataclass(frozen=True)
class GroundedExcerpt:
    """Provenance metadata for an excerpt verified against a source document span."""

    verbatim_text: str
    source_url: str
    locator: str
    span: SourceSpan
    match_start: int
    match_end: int


def normalize_with_char_map(text: str) -> tuple[str, list[int]]:
    """Normalize text by collapsing whitespace and casefolding, mapping indices to original text."""
    normalized_chars: list[str] = []
    norm_to_orig: list[int] = []

    for orig_idx, char in enumerate(text):
        if char.isspace():
            if not normalized_chars or normalized_chars[-1] == " ":
                continue
            normalized_chars.append(" ")
            norm_to_orig.append(orig_idx)
        else:
            lowered = char.casefold()
            for c in lowered:
                normalized_chars.append(c)
                norm_to_orig.append(orig_idx)

    # Pop trailing whitespace
    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        norm_to_orig.pop()

    return "".join(normalized_chars), norm_to_orig


def normalize_text_only(text: str) -> str:
    """Collapse whitespace and casefold a query string without tracking character positions."""
    norm_str, _ = normalize_with_char_map(text)
    return norm_str


class GroundingValidator:
    """Validates and locates source excerpts strictly within individual document spans."""

    def __init__(self, spans: list[SourceSpan], min_excerpt_chars: int = 8) -> None:
        self.min_excerpt_chars = min_excerpt_chars
        self.spans = spans

        # Precompute normalized representation and index map for each span
        self._indexed_spans: list[tuple[SourceSpan, str, list[int]]] = []
        for span in spans:
            norm_text, char_map = normalize_with_char_map(span.text)
            self._indexed_spans.append((span, norm_text, char_map))

    def locate(self, excerpt: str) -> GroundedExcerpt | None:
        """Locate an excerpt verbatim within a single span, or return None if ungrounded.

        Guarantees:
        - Must be at least min_excerpt_chars (after normalization).
        - Must match within a single span (no cross-span splicing).
        - Exact match after whitespace-collapsing and casefolding only (no fuzzy matching).
        - Recovers the exact original verbatim substring with refined locators.
        """
        if not excerpt:
            return None

        norm_excerpt = normalize_text_only(excerpt)
        if len(norm_excerpt) < self.min_excerpt_chars:
            return None

        for span, norm_text, char_map in self._indexed_spans:
            pos = norm_text.find(norm_excerpt)
            if pos != -1:
                start_norm = pos
                end_norm = pos + len(norm_excerpt)
                last_norm = end_norm - 1

                orig_start = char_map[start_norm]
                orig_end = char_map[last_norm] + 1

                verbatim_text = span.text[orig_start:orig_end]
                refined_locator = f"{span.locator};excerpt_chars={orig_start}-{orig_end}"

                return GroundedExcerpt(
                    verbatim_text=verbatim_text,
                    source_url=span.source_url,
                    locator=refined_locator,
                    span=span,
                    match_start=orig_start,
                    match_end=orig_end,
                )

        return None

    def is_grounded(self, excerpt: str) -> bool:
        """Return True if the excerpt can be located verbatim in at least one document span."""
        return self.locate(excerpt) is not None
