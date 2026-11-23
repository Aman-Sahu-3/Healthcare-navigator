"""Render metrics as a Markdown report (paste-ready for the README)."""

from __future__ import annotations

from app.eval.runner import ItemResult


def _fmt(v: object, suffix: str = "") -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}" if not suffix else f"{v:.1f}{suffix}"
    return f"{v}{suffix}"


def render_report(metrics: dict, results: list[ItemResult], meta: dict) -> str:
    c, r, g = metrics["counts"], metrics["retrieval"], metrics["generation"]
    k = r["k"]
    lines = [
        "# Evaluation report",
        "",
        f"- **Run:** {meta.get('timestamp', '')}",
        f"- **LLM:** `{meta.get('model', '')}` | "
        f"**Embeddings:** `{meta.get('embedding_model', '')}` | **top-k:** {k} | "
        f"**min_similarity:** {meta.get('min_similarity', '')}",
        f"- **Questions:** {c['total']} ({c['answerable']} answerable, "
        f"{c['unanswerable']} unanswerable) | **errors:** {c['errors']} | "
        f"**judged:** {c['judged']}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval: correct source in top-{k} | {_fmt(r.get(f'source_hit@{k}_pct'), '%')} |",
        f"| Retrieval: correct source at rank 1 | {_fmt(r.get('source_hit@1_pct'), '%')} |",
        f"| Retrieval: correct page in top-{k} | {_fmt(r.get(f'page_hit@{k}_pct'), '%')} |",
        f"| Retrieval: MRR | {_fmt(r.get('mrr'))} |",
        f"| Answered when answerable | {_fmt(g['answered_when_answerable_pct'], '%')} |",
        f"| Refused when unanswerable | {_fmt(g['refused_when_unanswerable_pct'], '%')} |",
        f"| Citation accuracy (cites expected doc) | {_fmt(g['citation_accuracy_pct'], '%')} |",
        f"| Faithfulness (LLM judge, 0–1) | {_fmt(g['faithfulness'])} |",
        f"| Correctness vs reference (LLM judge, 0–1) | {_fmt(g['correctness'])} |",
        f"| Latency p50 / p95 | {_fmt(metrics['latency_ms']['p50'])} / "
        f"{_fmt(metrics['latency_ms']['p95'])} ms |",
        "",
        "## Confidence calibration (answerable questions)",
        "",
        "| Level | Count | Mean correctness |",
        "|---|---|---|",
    ]
    for level, row in metrics["calibration"].items():
        lines.append(f"| {level} | {row['count']} | {_fmt(row['correctness'])} |")
    lines += [
        "",
        "Well calibrated = correctness drops from HIGH → MEDIUM → LOW.",
        "",
        "## Similarity distribution (top-1 cosine)",
        "",
        "| Set | min | median | max |",
        "|---|---|---|---|",
    ]
    for name, key in (("Answerable", "answerable_top1"), ("Unanswerable", "unanswerable_top1")):
        s = metrics["similarity"][key]
        lines.append(f"| {name} | {_fmt(s['min'])} | {_fmt(s['median'])} | {_fmt(s['max'])} |")

    sug = metrics["suggestions"]
    lines += ["", "## Suggested settings", ""]
    if sug["floor"]:
        f = sug["floor"]
        lines.append(
            f"- `MIN_SIMILARITY={f['min_similarity']}` — keeps {f['answerable_kept_pct']}% of "
            f"answerable questions, blocks {f['unanswerable_blocked_pct']}% of unanswerable "
            "ones before the LLM is called."
        )
    if sug["high_conf_threshold"]:
        lines.append(
            f"- `HIGH_CONF_THRESHOLD={sug['high_conf_threshold']}` (median of correct answers)"
        )
        lines.append(
            f"- `MEDIUM_CONF_THRESHOLD={sug['medium_conf_threshold']}` (10th percentile of correct)"
        )
    if not sug["floor"] and not sug["high_conf_threshold"]:
        lines.append("- Not enough data (need answerable + unanswerable questions and ≥5 judged).")

    failures = [x for x in results if x.error or not x.refusal_correct or x.cited_expected is False]
    if failures:
        lines += ["", "## Failures to review", "", "| id | issue | question |", "|---|---|---|"]
        for x in failures[:25]:
            if x.error:
                issue = "error"
            elif not x.refusal_correct:
                issue = "refused answerable" if x.answerable else "answered unanswerable"
            else:
                issue = "cited wrong doc"
            q = x.question.replace("|", "/")[:80]
            lines.append(f"| {x.id} | {issue} | {q} |")
    return "\n".join(lines) + "\n"
