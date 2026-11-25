"""Export the vector index to one .tar.gz file, and restore it from a URL at startup.

Why: free hosts (e.g. Render free) have no persistent disk and a tiny CPU. Instead of
re-embedding thousands of passages on every cold start, the server downloads a
prebuilt index (e.g. a GitHub Release asset) in a few seconds.

    python -m app.ingestion --export index.tar.gz      # build machine
    INDEX_URL=https://.../index.tar.gz                  # server env
    INDEX_SHA256=<printed by --export>                  # optional integrity check
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB safety cap


class SnapshotError(RuntimeError):
    pass


@dataclass
class ExportResult:
    path: Path
    sha256: str
    size_bytes: int
    manifest: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def has_index(chroma_dir: Path) -> bool:
    return (Path(chroma_dir) / "chroma.sqlite3").exists()


def export_index(chroma_dir: Path, out_path: Path, manifest: dict) -> ExportResult:
    """Pack chroma_dir (+ a manifest) into a .tar.gz. Run when no ingestion is in progress."""
    chroma_dir, out_path = Path(chroma_dir), Path(out_path)
    if not has_index(chroma_dir):
        raise SnapshotError(f"No index found in {chroma_dir}. Ingest documents first.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "created_at": datetime.now(UTC).isoformat(), "format": 1}

    with tarfile.open(out_path, "w:gz") as tar:
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
        for p in sorted(chroma_dir.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=f"chroma/{p.relative_to(chroma_dir).as_posix()}")
    return ExportResult(out_path, _sha256(out_path), out_path.stat().st_size, manifest)


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Reject absolute paths, '..', links and devices (tar path-traversal attacks)."""
    members = []
    for m in tar.getmembers():
        path = PurePosixPath(m.name)
        if path.is_absolute() or ".." in path.parts:
            raise SnapshotError(f"Unsafe path in snapshot: {m.name}")
        if not (m.isfile() or m.isdir()):
            raise SnapshotError(f"Unsupported entry in snapshot: {m.name}")
        members.append(m)
    return members


def _download(url: str, dest: Path, timeout: float) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "healthcare-navigator"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
        total = 0
        while block := resp.read(1 << 20):
            total += len(block)
            if total > MAX_DOWNLOAD_BYTES:
                raise SnapshotError("Snapshot exceeds the 1 GB download limit")
            f.write(block)


def restore_index(
    url: str,
    chroma_dir: Path,
    expected_sha256: str | None = None,
    expected_model: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Download a snapshot and unpack it into chroma_dir. Returns its manifest."""
    chroma_dir = Path(chroma_dir)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        archive = tmp / "index.tar.gz"
        _download(url, archive, timeout)

        if expected_sha256 and _sha256(archive) != expected_sha256.strip().lower():
            raise SnapshotError("Snapshot checksum mismatch (INDEX_SHA256)")

        with tarfile.open(archive, "r:gz") as tar:
            members = _safe_members(tar)
            names = {m.name for m in members}
            if MANIFEST not in names:
                raise SnapshotError("Snapshot has no manifest.json")
            manifest = json.load(tar.extractfile(MANIFEST))  # type: ignore[arg-type]
            if expected_model and manifest.get("embedding_model") != expected_model:
                raise SnapshotError(
                    f"Snapshot was built with {manifest.get('embedding_model')!r} but "
                    f"EMBEDDING_MODEL is {expected_model!r}; vectors would not match."
                )
            staging = tmp / "unpacked"
            for m in members:
                if m.name.startswith("chroma/"):
                    target = staging / m.name
                    if m.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, target.open("wb") as dst:  # type: ignore[union-attr]
                        shutil.copyfileobj(src, dst)

        if not has_index(staging / "chroma"):
            raise SnapshotError("Snapshot does not contain a Chroma index")
        chroma_dir.mkdir(parents=True, exist_ok=True)
        for item in (staging / "chroma").iterdir():
            dest = chroma_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), dest)

    logger.info(
        "Restored index snapshot (%s chunks, built %s) in %.1fs",
        manifest.get("chunks", "?"),
        manifest.get("created_at", "?"),
        time.perf_counter() - start,
    )
    return manifest
