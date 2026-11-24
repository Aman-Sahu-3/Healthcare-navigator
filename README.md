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
| `backend/app/ingestion` | PDF parsing, chunking, embedding, vector storage, retrieval CLI |
| `backend/app/rag` | LCEL chain, grounded prompt, citation mapping, confidence scoring |
| `backend/app/api` | FastAPI routes, errors, rate limiting, request IDs |
| `backend/app/schemas` | Pydantic request/response models |
| `backend/tests` | pytest suite |
| `frontend/` | React app *(Phase 5)* |
| `data/raw/` | Source PDFs (git-ignored) |
| `eval/` | Golden Q&A set (`golden.jsonl`) and evaluation reports |
| `backend/app/eval` | Evaluation harness: question generation, metrics, LLM judge, reports |

## Quick start

```bash
cp .env.example .env          # add your OPENAI_API_KEY
make install                  # venv + dependencies + pre-commit hooks
make check                    # lint + tests
make run                      # http://localhost:8000/health  and  /docs
```

On Windows without `make`, run the commands inside the Makefile manually from `backend/`.

## Ingestion (Phase 1)

```bash
# put PDFs in data/raw/, then from backend/ with the venv active:
python -m app.ingestion                       # ingest (skips unchanged files)
python -m app.ingestion --list                # what's stored
python -m app.ingestion --search "first-line treatment for hypertension"
python -m app.ingestion --force               # re-embed everything
python -m app.ingestion --reset               # wipe collection, then ingest
```

| Step | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF, per page | Fast, accurate; per-page docs give exact page citations |
| Cleaning | De-hyphenation, drop page-number lines, whitespace normalization | Cleaner chunks embed better |
| Chunking | Recursive splitter, 1000 chars / 150 overlap, per page | Respects paragraph → sentence boundaries; every chunk maps to one page |
| IDs | `<file-hash>:p<page>:c<n>` | Deterministic → re-ingestion never duplicates |
| Embeddings | `BAAI/bge-small-en-v1.5`, normalized | Strong retrieval for its size, runs on CPU |
| Vector DB | ChromaDB, persistent, cosine space | Similarity = 1 − cosine distance, range 0–1 |

## RAG chain (Phase 2)

```bash
python -m app.rag "What is the first-line treatment for hypertension?"
python -m app.rag "..." -k 8 --json
```

```
question -> top-k cosine retrieval (drop < MIN_SIMILARITY)
         -> nothing relevant? -> "Insufficient evidence" (no LLM call)
         -> numbered passages [1..n] -> grounded prompt -> LLM
         -> [n] markers mapped to source/page from metadata (invalid markers removed)
         -> confidence: HIGH / MEDIUM / LOW
```

**Anti-hallucination measures:** context-only prompt with mandatory `[n]` citations, an explicit
"Insufficient evidence" escape hatch, temperature 0, retrieval similarity floor, and citations
built from vector-store metadata rather than model output.

**Confidence rules** (thresholds in `.env`):

| Level | Rule |
|---|---|
| HIGH | best cited similarity ≥ 0.75 **and** ≥ 2 passages cited |
| MEDIUM | best cited similarity ≥ 0.60 |
| LOW | otherwise, or no valid citations, or model reported insufficient evidence |

## Evaluation (Phase 3)

```bash
python -m app.eval generate --n 30     # draft questions from your PDFs -> eval/golden_draft.jsonl
# review & edit, save as eval/golden.jsonl
python -m app.eval run                 # -> eval/results/report-<time>.md
```

| Metric | What it measures |
|---|---|
| Source hit@k / hit@1, MRR | Does retrieval surface the document that holds the answer? |
| Page hit@k | …and the right page? |
| Answered when answerable / Refused when unanswerable | Does the assistant answer real questions and refuse out-of-scope ones? |
| Citation accuracy | Does the answer cite the expected document? |
| Faithfulness (LLM judge) | Is every claim supported by the retrieved passages? |
| Correctness (LLM judge) | Does the answer match the reference answer? |
| Confidence calibration | Is HIGH actually more correct than MEDIUM and LOW? |

The run also suggests `MIN_SIMILARITY` and confidence thresholds from the observed similarity
distributions of answerable vs. unanswerable questions.

<!-- Paste the headline table from your latest eval/results/report-*.md here. -->

## REST API (Phase 4)

```bash
cd backend
uvicorn app.main:app --reload --reload-dir app --port 8000   # docs: http://localhost:8000/docs
```

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/query` | – (rate limited) | `{question, top_k?}` → answer, citations, confidence |
| GET | `/sources` | – | Indexed documents with page and chunk counts |
| GET | `/health` | – | Status (`ok` / `degraded`), LLM + index readiness |
| POST | `/ingest` | `X-API-Key` | Re-scan `data/raw` (`{"force": true}` to re-embed) |
| POST | `/ingest/upload` | `X-API-Key` | Upload and ingest one PDF |

Example:

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the first-line treatment for hypertension?"}'
```

```json
{
  "answer": "Thiazide diuretics, ACE inhibitors, ARBs or CCBs are first-line [1].",
  "citations": [{"id": 1, "source": "who_htn.pdf", "title": "...", "page": 12,
                 "snippet": "...", "similarity": 0.82}],
  "confidence": {"level": "HIGH", "score": 0.87, "reasons": ["..."]},
  "insufficient_evidence": false,
  "latency_ms": 1140
}
```

Every error has the same shape — `{"error": {"code", "message", "details?"}, "request_id"}` —
with LLM provider failures mapped to clear codes (`llm_rate_limited` 503, `llm_auth_error` 502,
`llm_timeout` 504). Models load once at startup; question text is never logged.

## Roadmap

- [x] Phase 0 — Foundations (repo, config, logging, CI)
- [x] Phase 1 — Ingestion pipeline
- [x] Phase 2 — RAG chain + confidence scoring
- [x] Phase 3 — Evaluation
- [x] Phase 4 — REST API
- [ ] Phase 5 — React frontend
- [ ] Phase 6 — Docker
- [ ] Phase 7 — Deployment + CI/CD
- [ ] Phase 8 — Hardening
