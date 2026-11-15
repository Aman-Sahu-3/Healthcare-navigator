"""PDF parsing: one LangChain Document per page, with citation-ready metadata.

Metadata on every page (and later every chunk):
    source      file name, e.g. "who_hypertension_2021.pdf"   -> shown on citation cards
    page        1-based page number                            -> shown on citation cards
    title       PDF metadata title, or a cleaned-up file name
    total_pages page count of the document
    file_hash   SHA-256 of the file bytes                      -> idempotent re-ingestion
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "treat-\nment" -> "treatment"
_PAGE_NUM_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t ]{2,}")


class PDFLoadError(Exception):
    """Raised when a file cannot be opened or parsed as a PDF."""


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphens
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _PAGE_NUM_LINE.match(ln)]
    text = "\n".join(lines)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _title_from(meta_title: str | None, path: Path) -> str:
    if meta_title and len(meta_title.strip()) > 3:
        return meta_title.strip()
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


def load_pdf(path: Path) -> list[Document]:
    """Parse a PDF into per-page Documents. Pages with no extractable text are skipped."""
    path = Path(path)
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # pymupdf raises several exception types
        raise PDFLoadError(f"Cannot open {path.name}: {exc}") from exc

    file_hash = file_sha256(path)
    with pdf:
        if pdf.needs_pass:
            raise PDFLoadError(f"{path.name} is password-protected")
        title = _title_from(pdf.metadata.get("title") if pdf.metadata else None, path)
        total = pdf.page_count
        docs: list[Document] = []
        for i, page in enumerate(pdf):
            text = clean_text(page.get_text("text", sort=True))
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": i + 1,
                        "title": title,
                        "total_pages": total,
                        "file_hash": file_hash,
                    },
                )
            )

    if not docs:
        logger.warning("%s: no extractable text (scanned PDF? needs OCR — skipped)", path.name)
    return docs
