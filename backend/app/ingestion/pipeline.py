"""Ingestion orchestration: PDFs -> pages -> chunks -> embeddings -> ChromaDB.

Re-running is safe:
  * unchanged file (same SHA-256)  -> skipped
  * changed file (same name, new hash) -> old chunks deleted, new ones added
  * --force -> everything re-embedded
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.ingestion.embeddings import get_embeddings
from app.ingestion.loader import PDFLoadError, file_sha256, load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.vectorstore import delete_source, get_vectorstore, has_file_hash

logger = logging.getLogger(__name__)

ADD_BATCH = 256  # well under Chroma's max batch size


@dataclass
class FileResult:
    source: str
    status: str  # "ingested" | "skipped" | "replaced" | "failed" | "empty"
    pages: int = 0
    chunks: int = 0
    error: str | None = None


@dataclass
class IngestReport:
    files: list[FileResult] = field(default_factory=list)
    seconds: float = 0.0

    def _n(self, status: str) -> int:
        return sum(1 for f in self.files if f.status == status)

    @property
    def total_chunks(self) -> int:
        return sum(f.chunks for f in self.files)

    def summary(self) -> str:
        return (
            f"{len(self.files)} files | ingested {self._n('ingested')} | "
            f"replaced {self._n('replaced')} | skipped {self._n('skipped')} | "
            f"empty {self._n('empty')} | failed {self._n('failed')} | "
            f"{self.total_chunks} new chunks | {self.seconds:.1f}s"
        )


def discover_pdfs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def build_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    emb = embeddings or get_embeddings(
        settings.embedding_model, settings.embedding_device, settings.embedding_batch_size
    )
    return get_vectorstore(emb, settings.chroma_dir, settings.chroma_collection)


def ingest_file(path: Path, store: Chroma, settings: Settings, force: bool = False) -> FileResult:
    name = path.name
    try:
        file_hash = file_sha256(path)
        if not force and has_file_hash(store, file_hash):
            return FileResult(name, "skipped")

        removed = delete_source(store, name)  # stale version of this file, if any
        pages = load_pdf(path)
        if not pages:
            return FileResult(name, "empty")

        chunks = split_documents(
            pages, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars
        )
        if not chunks:  # every fragment was below min_chunk_chars
            return FileResult(name, "empty", pages=len(pages))
        for i in range(0, len(chunks), ADD_BATCH):
            batch = chunks[i : i + ADD_BATCH]
            store.add_documents(batch, ids=[c.id for c in batch])

        status = "replaced" if removed and not force else "ingested"
        return FileResult(name, status, pages=len(pages), chunks=len(chunks))
    except PDFLoadError as exc:
        return FileResult(name, "failed", error=str(exc))
    except Exception as exc:  # keep going; one bad file shouldn't stop the batch
        logger.exception("Unexpected error ingesting %s", name)
        return FileResult(name, "failed", error=f"{type(exc).__name__}: {exc}")


def ingest_path(
    path: Path,
    settings: Settings,
    embeddings: Embeddings | None = None,
    force: bool = False,
) -> IngestReport:
    start = time.perf_counter()
    report = IngestReport()
    pdfs = discover_pdfs(path)
    if not pdfs:
        logger.warning("No PDFs found under %s", path)
        return report

    store = build_store(settings, embeddings)
    for n, pdf in enumerate(pdfs, 1):
        result = ingest_file(pdf, store, settings, force=force)
        report.files.append(result)
        extra = f" ({result.pages} pages, {result.chunks} chunks)" if result.chunks else ""
        err = f" — {result.error}" if result.error else ""
        logger.info("[%d/%d] %-9s %s%s%s", n, len(pdfs), result.status, pdf.name, extra, err)

    report.seconds = time.perf_counter() - start
    return report
