"""Resume reader for PDF, plain text, and Markdown with verbatim span extraction and exact locators."""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pypdf import PdfReader

from pilot.ingestion.cache import IngestionCache


# ============================================================================
# Models & Errors
# ============================================================================
class SourceSpan(BaseModel):
    """An atomic, verbatim snippet from a document with exact provenance."""

    text: str = Field(description="Exact verbatim excerpt from source")
    locator: str = Field(description="Exact locator, e.g. page=1;chars=100-240 or line=14-16")
    source_url: str = Field(description="Absolute file path or file:// URI")


class ResumeDocument(BaseModel):
    """Parsed structured resume document with full text and atomic spans."""

    source_url: str
    content_type: Literal["pdf", "text", "markdown"]
    raw_text: str
    spans: list[SourceSpan]
    fetched_at: datetime


class ResumeReaderError(Exception):
    """Base error for resume parsing failures."""


class EncryptedPDFError(ResumeReaderError):
    """Raised when a PDF is password-protected or encrypted."""


class ImageOnlyPDFError(ResumeReaderError):
    """Raised when a PDF contains no or negligible extractable text (scanned/image-only)."""


class UnsupportedFileFormatError(ResumeReaderError):
    """Raised when an unsupported file type is provided."""


# ============================================================================
# Reader Implementation
# ============================================================================
class ResumeReader:
    """Extracts raw text and atomic spans from resumes (PDF, TXT, MD)."""

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.cache = IngestionCache(cache_dir=cache_dir)

    def read(self, file_path: str | Path, force_refresh: bool = False) -> ResumeDocument:
        """Parse resume file into a ResumeDocument with verbatim spans."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume file not found at {path}")

        # Check cache by file content hash
        file_bytes = path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_key = f"{path.as_posix()}:{file_hash}"

        if not force_refresh:
            cached = self.cache.get("resumes", cache_key)
            if cached:
                return ResumeDocument.model_validate(cached)

        suffix = path.suffix.lower()
        source_url = path.as_uri()

        if suffix == ".pdf":
            doc = self._parse_pdf(path, source_url, file_bytes)
        elif suffix in [".txt", ".text"]:
            doc = self._parse_text(
                path, source_url, file_bytes.decode("utf-8", errors="replace"), "text"
            )
        elif suffix in [".md", ".markdown"]:
            doc = self._parse_text(
                path, source_url, file_bytes.decode("utf-8", errors="replace"), "markdown"
            )
        else:
            raise UnsupportedFileFormatError(
                f"Unsupported resume file extension '{suffix}'. Expected .pdf, .txt, or .md."
            )

        # Cache document
        self.cache.set("resumes", cache_key, doc.model_dump(mode="json"))
        return doc

    def _parse_pdf(self, path: Path, source_url: str, file_bytes: bytes) -> ResumeDocument:
        """Extract pages and atomic spans from PDF documents."""
        try:
            reader = PdfReader(path)
        except Exception as e:
            raise ResumeReaderError(f"Failed to open PDF: {e}") from e

        if reader.is_encrypted:
            try:
                # Attempt decryption with empty password for unencrypted permissions
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise EncryptedPDFError(f"PDF at {path} is encrypted and password-protected.")
            except Exception as e:
                raise EncryptedPDFError(f"PDF at {path} is encrypted: {e}") from e

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ImageOnlyPDFError(f"PDF at {path} contains zero pages.")

        page_texts: list[str] = []
        spans: list[SourceSpan] = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            extracted = page.extract_text() or ""
            page_texts.append(extracted)

            page_spans = self._extract_spans_from_block(
                text_block=extracted,
                locator_prefix=f"page={page_num}",
                source_url=source_url,
            )
            spans.extend(page_spans)

        raw_text = "\n\n".join(page_texts)

        # Scanned / Image-only detection: check total extracted text density
        non_whitespace_chars = len(re.sub(r"\s+", "", raw_text))
        min_expected_chars = max(40, num_pages * 25)
        if non_whitespace_chars < min_expected_chars:
            raise ImageOnlyPDFError(
                f"PDF at {path} has insufficient extractable text ({non_whitespace_chars} chars across "
                f"{num_pages} pages). Scanned or image-only PDFs are not supported without OCR."
            )

        return ResumeDocument(
            source_url=source_url,
            content_type="pdf",
            raw_text=raw_text,
            spans=spans,
            fetched_at=datetime.now(UTC),
        )

    def _parse_text(
        self,
        path: Path,
        source_url: str,
        text_content: str,
        content_type: Literal["text", "markdown"],
    ) -> ResumeDocument:
        """Extract atomic spans from plain text or markdown files."""
        lines = text_content.splitlines(keepends=True)
        spans: list[SourceSpan] = []

        bullet_pattern = re.compile(r"^\s*([*\-•–]|\d+\.)\s+")

        current_span_lines: list[str] = []
        span_start_line = 1

        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not stripped:
                # Paragraph break
                if current_span_lines:
                    span_text = "".join(current_span_lines).strip()
                    if span_text:
                        end_line = line_idx - 1
                        locator = (
                            f"line={span_start_line}"
                            if span_start_line == end_line
                            else f"line={span_start_line}-{end_line}"
                        )
                        spans.append(
                            SourceSpan(text=span_text, locator=locator, source_url=source_url)
                        )
                    current_span_lines = []
                span_start_line = line_idx + 1
                continue

            if bullet_pattern.match(line):
                # New bullet boundary
                if current_span_lines:
                    span_text = "".join(current_span_lines).strip()
                    if span_text:
                        end_line = line_idx - 1
                        locator = (
                            f"line={span_start_line}"
                            if span_start_line == end_line
                            else f"line={span_start_line}-{end_line}"
                        )
                        spans.append(
                            SourceSpan(text=span_text, locator=locator, source_url=source_url)
                        )
                    current_span_lines = []
                    span_start_line = line_idx

                current_span_lines.append(line)
            else:
                if not current_span_lines:
                    span_start_line = line_idx
                current_span_lines.append(line)

        if current_span_lines:
            span_text = "".join(current_span_lines).strip()
            if span_text:
                end_line = len(lines)
                locator = (
                    f"line={span_start_line}"
                    if span_start_line == end_line
                    else f"line={span_start_line}-{end_line}"
                )
                spans.append(SourceSpan(text=span_text, locator=locator, source_url=source_url))

        return ResumeDocument(
            source_url=source_url,
            content_type=content_type,
            raw_text=text_content,
            spans=spans,
            fetched_at=datetime.now(UTC),
        )

    def _extract_spans_from_block(
        self,
        text_block: str,
        locator_prefix: str,
        source_url: str,
    ) -> list[SourceSpan]:
        """Split a page/block into paragraph and bullet spans with character offset locators."""
        spans: list[SourceSpan] = []
        if not text_block.strip():
            return spans

        # Split into paragraph / bullet chunks while tracking char offsets
        lines = text_block.splitlines(keepends=True)
        bullet_pattern = re.compile(r"^\s*([*\-•–]|\d+\.)\s+")

        current_lines: list[str] = []
        current_start_offset = 0
        running_offset = 0

        for line in lines:
            stripped = line.strip()
            line_len = len(line)

            if not stripped:
                if current_lines:
                    raw_span = "".join(current_lines)
                    span_text = raw_span.strip()
                    if span_text:
                        start_char = current_start_offset + raw_span.find(span_text)
                        end_char = start_char + len(span_text)
                        locator = f"{locator_prefix};chars={start_char}-{end_char}"
                        spans.append(
                            SourceSpan(text=span_text, locator=locator, source_url=source_url)
                        )
                    current_lines = []
                running_offset += line_len
                current_start_offset = running_offset
                continue

            if bullet_pattern.match(line):
                if current_lines:
                    raw_span = "".join(current_lines)
                    span_text = raw_span.strip()
                    if span_text:
                        start_char = current_start_offset + raw_span.find(span_text)
                        end_char = start_char + len(span_text)
                        locator = f"{locator_prefix};chars={start_char}-{end_char}"
                        spans.append(
                            SourceSpan(text=span_text, locator=locator, source_url=source_url)
                        )
                    current_lines = []
                    current_start_offset = running_offset

                current_lines.append(line)
            else:
                current_lines.append(line)

            running_offset += line_len

        if current_lines:
            raw_span = "".join(current_lines)
            span_text = raw_span.strip()
            if span_text:
                start_char = current_start_offset + raw_span.find(span_text)
                end_char = start_char + len(span_text)
                locator = f"{locator_prefix};chars={start_char}-{end_char}"
                spans.append(SourceSpan(text=span_text, locator=locator, source_url=source_url))

        return spans
