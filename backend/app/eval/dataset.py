"""Golden evaluation set: one JSON object per line (JSONL).

Example lines:
  {"id": "q001", "question": "What is first-line therapy for hypertension?",
   "answerable": true, "expected_sources": ["who_htn_2021.pdf"], "expected_pages": [12],
   "reference_answer": "Thiazide diuretics, ACE inhibitors, ARBs or CCBs."}
  {"id": "neg01", "question": "What is the capital of France?", "answerable": false}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    id: str
    question: str = Field(min_length=3)
    answerable: bool = True
    expected_sources: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    reference_answer: str | None = None
    origin: Literal["manual", "generated", "negative"] = "manual"


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: invalid golden item: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return items


def save_golden(items: list[GoldenItem], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(exclude_defaults=False), ensure_ascii=False) + "\n")
