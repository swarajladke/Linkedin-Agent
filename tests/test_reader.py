"""Tests for resume reader parsing, verbatim span extraction, locators, and error handling."""

from pathlib import Path

import pytest

from pilot.ingestion.reader import (
    ImageOnlyPDFError,
    ResumeDocument,
    ResumeReader,
    UnsupportedFileFormatError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_read_plain_text_resume(tmp_path):
    """Test reading a plain text resume with line-based locators and verbatim spans."""
    reader = ResumeReader(cache_dir=tmp_path / ".cache")
    txt_path = FIXTURES_DIR / "sample_resume.txt"

    doc = reader.read(txt_path)

    assert isinstance(doc, ResumeDocument)
    assert doc.content_type == "text"
    assert doc.source_url.startswith("file://")
    assert len(doc.spans) > 0

    # Assert every span text appears verbatim in raw_text
    for span in doc.spans:
        assert span.text in doc.raw_text
        assert span.locator.startswith("line=")
        assert span.source_url == doc.source_url

    # Check caching
    cached_doc = reader.read(txt_path)
    assert cached_doc.raw_text == doc.raw_text
    assert len(cached_doc.spans) == len(doc.spans)


def test_read_pdf_resume(tmp_path):
    """Test reading a PDF resume with page and char-based locators."""
    reader = ResumeReader(cache_dir=tmp_path / ".cache")
    pdf_path = FIXTURES_DIR / "sample_resume.pdf"

    doc = reader.read(pdf_path)

    assert isinstance(doc, ResumeDocument)
    assert doc.content_type == "pdf"
    assert "Alex Candidate" in doc.raw_text
    assert len(doc.spans) > 0

    for span in doc.spans:
        assert span.text in doc.raw_text
        assert span.locator.startswith("page=1;chars=")
        assert span.source_url == doc.source_url


def test_image_only_pdf_raises_error(tmp_path):
    """Test that a scanned or image-only PDF with negligible text raises ImageOnlyPDFError."""
    reader = ResumeReader(cache_dir=tmp_path / ".cache")
    image_pdf_path = FIXTURES_DIR / "image_only_resume.pdf"

    with pytest.raises(ImageOnlyPDFError) as exc_info:
        reader.read(image_pdf_path)

    assert "insufficient extractable text" in str(exc_info.value) or "zero pages" in str(
        exc_info.value
    )


def test_unsupported_file_format_raises_error(tmp_path):
    """Test that unsupported file formats raise UnsupportedFileFormatError."""
    reader = ResumeReader(cache_dir=tmp_path / ".cache")
    fake_doc = tmp_path / "resume.docx"
    fake_doc.write_text("Hello docx", encoding="utf-8")

    with pytest.raises(UnsupportedFileFormatError):
        reader.read(fake_doc)
