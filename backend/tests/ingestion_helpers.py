"""Test helpers: synthetic PDFs and a tiny offline embedder."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pymupdf
from langchain_core.embeddings import Embeddings


def make_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed vector, L2-normalized.

    Not semantic, but texts that share words get high cosine similarity — enough to
    verify retrieval wiring offline and deterministically.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) < 3:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
