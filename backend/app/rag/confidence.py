"""Deterministic, explainable confidence scoring.

Level rules (thresholds come from settings):
    LOW    the model said "insufficient evidence", or cited nothing valid
    HIGH   best cited similarity >= high threshold AND >= 2 passages cited
    MEDIUM best cited similarity >= medium threshold
    LOW    otherwise

The numeric score (0..1) blends evidence strength and corroboration, for sorting/analytics:
    score = 0.7 * best_cited_similarity + 0.3 * min(cited_passages / 3, 1)
"""

from __future__ import annotations

from app.schemas.rag import Citation, Confidence


def score_confidence(
    citations: list[Citation],
    insufficient: bool,
    high_threshold: float,
    medium_threshold: float,
) -> Confidence:
    if insufficient:
        return Confidence(
            level="LOW", score=0.0, reasons=["Model reported insufficient evidence in sources"]
        )
    if not citations:
        return Confidence(
            level="LOW", score=0.0, reasons=["Answer is not supported by any cited passage"]
        )

    best = max(c.similarity for c in citations)
    n_cited = len(citations)
    n_docs = len({c.source for c in citations})
    score = round(max(0.0, min(1.0, 0.7 * best + 0.3 * min(n_cited / 3, 1.0))), 3)

    reasons = [
        f"Best cited passage similarity {best:.2f}",
        f"{n_cited} passage(s) cited from {n_docs} document(s)",
    ]
    if best >= high_threshold and n_cited >= 2:
        level = "HIGH"
    elif best >= medium_threshold:
        level = "MEDIUM"
        if best >= high_threshold:
            reasons.append("Only one supporting passage")
    else:
        level = "LOW"
        reasons.append(f"Similarity below {medium_threshold:.2f}")
    return Confidence(level=level, score=score, reasons=reasons)
