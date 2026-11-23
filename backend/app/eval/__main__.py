"""Evaluation CLI.

python -m app.eval generate --n 30        draft a golden set -> eval/golden_draft.jsonl
(review it, then save as eval/golden.jsonl)
python -m app.eval run                    evaluate -> eval/results/report-<time>.md
python -m app.eval run --no-judge         skip LLM-judge calls (half the API usage)
python -m app.eval run --limit 10 --sleep 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging


def _cmd_generate(args, settings) -> int:
    from app.eval.dataset import save_golden
    from app.eval.generate import generate_golden
    from app.ingestion.pipeline import build_store
    from app.rag.llm import get_llm

    store = build_store(settings)
    items = generate_golden(
        store,
        get_llm(settings),
        n=args.n,
        include_negatives=not args.no_negatives,
        sleep_s=args.sleep,
        min_chars=args.min_chars,
    )
    out = args.out or settings.eval_dir / "golden_draft.jsonl"
    save_golden(items, out)
    gen = sum(i.origin == "generated" for i in items)
    print(f"Wrote {len(items)} questions ({gen} generated, {len(items) - gen} negative) to {out}")
    print("Review them, then save as eval/golden.jsonl and run: python -m app.eval run")
    return 0


def _cmd_run(args, settings) -> int:
    from app.eval.dataset import load_golden
    from app.eval.metrics import compute_metrics
    from app.eval.report import render_report
    from app.eval.runner import run_eval
    from app.rag.chain import RAGService
    from app.rag.llm import get_llm

    golden = args.golden or settings.eval_dir / "golden.jsonl"
    if not Path(golden).exists():
        print(f"Golden set not found: {golden}\nCreate one with: python -m app.eval generate")
        return 2
    items = load_golden(golden)[: args.limit or None]
    service = RAGService.from_settings(settings)
    judge = None if args.no_judge else get_llm(settings)
    k = args.k or settings.retrieval_top_k

    results = run_eval(service, items, k=k, judge_llm=judge, sleep_s=args.sleep)
    metrics = compute_metrics(results, k)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    meta = {
        "timestamp": stamp,
        "model": service.model_name,
        "embedding_model": settings.embedding_model,
        "min_similarity": settings.min_similarity,
        "golden": str(golden),
    }
    out_dir = settings.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(metrics, results, meta)
    (out_dir / f"report-{stamp}.md").write_text(report, encoding="utf-8")
    (out_dir / f"run-{stamp}.json").write_text(
        json.dumps(
            {"meta": meta, "metrics": metrics, "results": [r.model_dump() for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + report)
    print(f"Saved: {out_dir / f'report-{stamp}.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="draft golden questions from your indexed PDFs")
    g.add_argument("--n", type=int, default=30)
    g.add_argument("--out", type=Path)
    g.add_argument("--no-negatives", action="store_true")
    g.add_argument("--sleep", type=float, default=0.0, help="seconds between LLM calls")
    g.add_argument("--min-chars", type=int, default=400, help="skip shorter chunks")

    r = sub.add_parser("run", help="evaluate the golden set")
    r.add_argument("--golden", type=Path)
    r.add_argument("-k", type=int)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--sleep", type=float, default=0.0, help="seconds between questions")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING")

    from app.rag.llm import LLMNotConfiguredError

    try:
        if args.cmd == "generate":
            return _cmd_generate(args, settings)
        return _cmd_run(args, settings)
    except LLMNotConfiguredError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
