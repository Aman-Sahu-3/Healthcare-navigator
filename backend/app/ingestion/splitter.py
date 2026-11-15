"""Recursive text chunking that preserves page metadata and assigns stable chunk IDs."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try to split on paragraphs, then lines, then sentences, then words.
_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        keep_separator="end",
        strip_whitespace=True,
    )


def split_documents(
    pages: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int = 0,
) -> list[Document]:
    """Split page Documents into chunks.

    Chunking happens per page so every chunk maps to exactly one page number,
    which keeps citations precise. Each chunk gets a deterministic ID:
    "<file_hash[:16]>:p<page>:c<index>" — re-ingesting the same file produces the same IDs.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for page in pages:
        pieces = splitter.split_text(page.page_content)
        idx = 0
        for piece in pieces:
            if len(piece) < min_chunk_chars:
                continue
            meta = dict(page.metadata)
            meta["chunk_index"] = idx
            meta["chunk_id"] = f"{meta['file_hash'][:16]}:p{meta['page']}:c{idx}"
            chunks.append(Document(page_content=piece, metadata=meta, id=meta["chunk_id"]))
            idx += 1
    return chunks
