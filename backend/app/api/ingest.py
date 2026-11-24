"""Admin-only ingestion endpoints (require the X-API-Key header)."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import Services, get_services, get_store, require_admin
from app.api.errors import APIError
from app.core.config import Settings, get_settings
from app.ingestion.pipeline import IngestReport, ingest_file, ingest_path
from app.schemas.api import ErrorResponse, IngestFileResult, IngestRequest, IngestResponse

router = APIRouter(
    prefix="/ingest",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
        409: {"model": ErrorResponse, "description": "Another ingestion is running"},
        503: {"model": ErrorResponse, "description": "Ingestion disabled"},
    },
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


def _to_response(report: IngestReport) -> IngestResponse:
    return IngestResponse(
        files=[IngestFileResult(**vars(f)) for f in report.files],
        new_chunks=report.total_chunks,
        seconds=round(report.seconds, 2),
        summary=report.summary(),
    )


class _IngestGuard:
    """Only one ingestion at a time — concurrent writes to Chroma waste work."""

    def __init__(self, services: Services):
        self.lock = services.ingest_lock

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            raise APIError(409, "ingest_in_progress", "Another ingestion is already running")

    def __exit__(self, *exc):
        self.lock.release()


@router.post("", response_model=IngestResponse, summary="Re-scan the documents folder")
def ingest_folder(
    body: IngestRequest | None = None,
    settings: Settings = Depends(get_settings),
    services: Services = Depends(get_services),
) -> IngestResponse:
    store = get_store(services)
    force = bool(body and body.force)
    with _IngestGuard(services):
        report = ingest_path(settings.raw_docs_dir, settings, force=force, store=store)
    return _to_response(report)


@router.post("/upload", response_model=IngestResponse, summary="Upload and ingest one PDF")
def ingest_upload(
    file: UploadFile = File(description="A text-based PDF"),
    settings: Settings = Depends(get_settings),
    services: Services = Depends(get_services),
) -> IngestResponse:
    store = get_store(services)
    name = _SAFE_NAME.sub("_", Path(file.filename or "").name).strip(" .")
    if not name.lower().endswith(".pdf"):
        raise APIError(400, "invalid_file", "Only .pdf files are accepted")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise APIError(413, "file_too_large", f"File exceeds {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise APIError(400, "invalid_file", "File is not a valid PDF")

    dest = Path(settings.raw_docs_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _IngestGuard(services):
        dest.write_bytes(data)
        result = ingest_file(dest, store, settings)
    report = IngestReport(files=[result])
    return _to_response(report)
