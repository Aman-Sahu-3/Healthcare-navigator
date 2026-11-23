"""Ask the RAG assistant from the command line.

python -m app.rag "What is first-line treatment for hypertension?"
python -m app.rag "..." -k 8
python -m app.rag "..." --json
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from app.core.config import get_settings
from app.core.logging import configure_logging

_BADGE = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=None, help="passages to retrieve")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")

    from app.rag.chain import RAGService
    from app.rag.llm import LLMNotConfiguredError

    try:
        service = RAGService.from_settings(settings)
        result = service.answer(args.question, k=args.k)
    except LLMNotConfiguredError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    conf = result.confidence
    print(f"\nConfidence: {_BADGE[conf.level]}  (score {conf.score:.2f})")
    for r in conf.reasons:
        print(f"  - {r}")
    print("\nAnswer:\n" + textwrap.indent(result.answer, "  "))
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.id}] {c.title} - p.{c.page}  ({c.source}, sim {c.similarity:.2f})")
    print(
        f"\n{result.retrieved_chunks} passages retrieved | {result.model} | {result.latency_ms} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
