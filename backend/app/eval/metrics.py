"""Aggregate metrics + data-driven threshold suggestions."""

from __future__ import annotations

from statistics import mean, median

from app.eval.judge import GRADE_SCORE
from app.eval.runner import ItemResult


def _pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def _avg(xs: list[float]) -> float | None:
    return round(mean(xs), 3) if xs else None


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def suggest_min_similarity(answerable: list[float], unanswerable: list[float]) -> dict | None:
    """Pick the similarity floor that best separates answerable from unanswerable questions.

    Maximises balanced accuracy: (answerable kept + unanswerable dropped) / 2.
    """
    if not answerable or not unanswerable:
        return None
    candidates = sorted(set(answerable) | set(unanswerable))
    best = None
    for t in candidates:
        kept = sum(s >= t for s in answerable) / len(answerable)
        dropped = sum(s < t for s in unanswerable) / len(unanswerable)
        score = (kept + dropped) / 2
        if best is None or score > best[0]:
            best = (score, t, kept, dropped)
    score, t, kept, dropped = best
    return {
        "min_similarity": round(t, 3),
        "balanced_accuracy": round(100 * score, 1),
        "answerable_kept_pct": round(100 * kept, 1),
        "unanswerable_blocked_pct": round(100 * dropped, 1),
    }


def compute_metrics(results: list[ItemResult], k: int) -> dict:
    ok = [r for r in results if not r.error]
    ans = [r for r in ok if r.answerable]
    neg = [r for r in ok if not r.answerable]
    expects = [r for r in ans if r.has_expected]

    judged = [r for r in ans if r.judgement and r.judgement.faithfulness]
    correct_judged = [r for r in ans if r.judgement and r.judgement.correctness]

    def corr(r: ItemResult) -> float:
        return GRADE_SCORE[r.judgement.correctness]  # type: ignore[index, union-attr]

    calibration = {}
    for level in ("HIGH", "MEDIUM", "LOW"):
        group = [r for r in ans if r.confidence == level]
        graded = [r for r in group if r.judgement and r.judgement.correctness]
        calibration[level] = {
            "count": len(group),
            "correctness": _avg([corr(r) for r in graded]),
        }

    lat = [float(r.latency_ms) for r in ok if r.latency_ms]
    ans_sims = [r.top_similarity for r in ans]
    neg_sims = [r.top_similarity for r in neg]
    correct_sims = [
        r.top_similarity
        for r in correct_judged
        if r.judgement.correctness == "yes"  # type: ignore[union-attr]
    ]

    return {
        "counts": {
            "total": len(results),
            "answerable": len(ans),
            "unanswerable": len(neg),
            "errors": len(results) - len(ok),
            "judged": len(judged),
        },
        "retrieval": {
            "k": k,
            f"source_hit@{k}_pct": _pct([r.source_rank is not None for r in expects]),
            "source_hit@1_pct": _pct([r.source_rank == 1 for r in expects]),
            f"page_hit@{k}_pct": _pct(
                [bool(r.page_hit) for r in expects if r.page_hit is not None]
            ),
            "mrr": _avg([1 / r.source_rank if r.source_rank else 0.0 for r in expects]),
        },
        "generation": {
            "answered_when_answerable_pct": _pct([not r.refused for r in ans]),
            "refused_when_unanswerable_pct": _pct([r.refused for r in neg]),
            "citation_accuracy_pct": _pct(
                [bool(r.cited_expected) for r in ans if r.cited_expected is not None]
            ),
            "faithfulness": _avg(
                [GRADE_SCORE[r.judgement.faithfulness] for r in judged]  # type: ignore[index, union-attr]
            ),
            "correctness": _avg([corr(r) for r in correct_judged]),
        },
        "calibration": calibration,
        "latency_ms": {
            "p50": _percentile(lat, 50),
            "p95": _percentile(lat, 95),
        },
        "similarity": {
            "answerable_top1": {
                "min": min(ans_sims, default=None),
                "median": round(median(ans_sims), 3) if ans_sims else None,
                "max": max(ans_sims, default=None),
            },
            "unanswerable_top1": {
                "min": min(neg_sims, default=None),
                "median": round(median(neg_sims), 3) if neg_sims else None,
                "max": max(neg_sims, default=None),
            },
        },
        "suggestions": {
            "floor": suggest_min_similarity(ans_sims, neg_sims),
            "high_conf_threshold": (
                round(_percentile(correct_sims, 50), 3) if len(correct_sims) >= 5 else None
            ),
            "medium_conf_threshold": (
                round(_percentile(correct_sims, 10), 3) if len(correct_sims) >= 5 else None
            ),
        },
    }
