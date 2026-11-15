# Healthcare Knowledge Navigator

A medical RAG assistant that answers clinical questions from guidelines and research papers,
with page-level source citations and a HIGH / MEDIUM / LOW confidence score.

> ⚠️ For informational use by healthcare professionals. Not a substitute for clinical judgment.

## Architecture (target)

```
PDFs ─► Ingestion (PyMuPDF → recursive chunking → HF embeddings) ─► ChromaDB
                                                                      │
React UI ─► FastAPI /query ─► LCEL RAG chain (top-k cosine retrieval ─┘
                               → grounded prompt → LLM → citations + confidence)
```

## Repository layout

| Path | Purpose |
|---|---|
| `backend/app/core` | Settings (env-driven) and logging |
| `backend/app/ingestion` | PDF parsing, chunking, embedding, vector storage *(Phase 1)* |
| `backend/app/rag` | Retriever, prompt, chain, confidence scoring *(Phase 2)* |
| `backend/app/api` | FastAPI routers |
| `backend/app/schemas` | Pydantic request/response models *(Phase 4)* |
| `backend/tests` | pytest suite |
| `frontend/` | React app *(Phase 5)* |
| `data/raw/` | Source PDFs (git-ignored) |
| `eval/` | Golden Q&A set and evaluation scripts *(Phase 3)* |

## Quick start

```bash
cp .env.example .env          # add your OPENAI_API_KEY
make install                  # venv + dependencies + pre-commit hooks
make check                    # lint + tests
make run                      # http://localhost:8000/health  and  /docs
```

On Windows without `make`, run the commands inside the Makefile manually from `backend/`.

## Roadmap

- [x] Phase 0 — Foundations (repo, config, logging, CI)
- [ ] Phase 1 — Ingestion pipeline
- [ ] Phase 2 — RAG chain + confidence scoring
- [ ] Phase 3 — Evaluation
- [ ] Phase 4 — REST API
- [ ] Phase 5 — React frontend
- [ ] Phase 6 — Docker
- [ ] Phase 7 — Deployment + CI/CD
- [ ] Phase 8 — Hardening
