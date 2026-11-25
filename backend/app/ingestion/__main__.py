"""Ingestion CLI.

python -m app.ingestion                 ingest everything in data/raw
python -m app.ingestion path/to/file.pdf
python -m app.ingestion --force         re-embed everything
python -m app.ingestion --reset         wipe the collection, then ingest
python -m app.ingestion --list          show what's stored
python -m app.ingestion --search "first-line treatment for hypertension" [-k 5]
python -m app.ingestion --export index.tar.gz   pack the index for deployment (INDEX_URL)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", type=Path, default=settings.raw_docs_dir)
    parser.add_argument("--force", action="store_true", help="re-embed even unchanged files")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--list", action="store_true", help="list stored documents and exit")
    parser.add_argument("--search", metavar="QUERY", help="run a retrieval test and exit")
    parser.add_argument("-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument(
        "--export", metavar="FILE", type=Path, help="write the index to a .tar.gz and exit"
    )
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)

    from app.ingestion.pipeline import build_store, ingest_path
    from app.ingestion.vectorstore import count, list_sources

    if args.export:
        from app.ingestion.snapshot import SnapshotError, export_index

        store = build_store(settings)
        manifest = {
            "embedding_model": settings.embedding_model,
            "collection": settings.chroma_collection,
            "chunks": count(store),
            "documents": len(list_sources(store)),
        }
        try:
            result = export_index(settings.chroma_dir, args.export, manifest)
        except SnapshotError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        print(
            f"Exported {manifest['documents']} documents / {manifest['chunks']} chunks\n"
            f"  file:   {result.path}  ({result.size_bytes / 1e6:.1f} MB)\n"
            f"  sha256: {result.sha256}\n"
            "Upload it (e.g. as a GitHub Release asset), then set INDEX_URL and INDEX_SHA256."
        )
        return 0

    if args.list or args.search:
        store = build_store(settings)
        if args.list:
            rows = list_sources(store)
            if not rows:
                print("Collection is empty. Run: python -m app.ingestion")
            for r in rows:
                print(f"{r['chunks']:>5} chunks  {r['total_pages'] or '?':>4} pages  {r['source']}")
            print(f"Total chunks: {count(store)}")
            return 0

        from app.ingestion.search import retrieve

        hits = retrieve(store, args.search, k=args.k)
        if not hits:
            print("No results. Is the collection empty?")
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] similarity={h.similarity:.3f}  {h.source}  p.{h.page}")
            print(textwrap.indent(textwrap.shorten(h.text, 400), "    "))
        return 0

    if args.reset:
        store = build_store(settings)
        store.delete_collection()
        print(f"Deleted collection '{settings.chroma_collection}'.")

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 2

    report = ingest_path(args.path, settings, force=args.force)
    print("\n" + report.summary())
    failed = [f for f in report.files if f.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
