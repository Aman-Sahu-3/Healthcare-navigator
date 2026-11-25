# Healthcare Knowledge Navigator

A medical RAG assistant that answers clinical questions from guidelines and research papers,
with page-level source citations and a HIGH / MEDIUM / LOW confidence score.

> ⚠️ For informational use by healthcare professionals. Not a substitute for clinical judgment.

## Architecture (target)

```
PDFs ─► Ingestion (PyMuPDF → recursive chunking → bge-small ONNX embeddings) ─► ChromaDB
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
| `frontend/` | React app (Vite, TypeScript, Tailwind, TanStack Query) |
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
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX), normalized | Strong retrieval for its size; no PyTorch, so the API fits in 512 MB |
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

## Frontend (Phase 5)

```bash
# terminal 1 — API
cd backend && uvicorn app.main:app --reload --reload-dir app --port 8000
# terminal 2 — UI
cd frontend && npm ci && npm run dev      # http://localhost:5173
```

- Question box with example prompts, passage-count option and keyboard submit
- Answer with inline numbered markers; clicking one jumps to and highlights its source card
- Citation cards: document title, file, **page number**, passage snippet, similarity bar
- Confidence badge — **green HIGH / amber MEDIUM / red LOW** — that explains its reasoning
- Distinct "no supported answer" state, friendly error messages (rate limits, server down)
- Recent-question history (stored only in the browser), indexed-document list, live server status
- Persistent "not medical advice" banner; answers rendered without injecting model HTML

<!-- Add a screenshot: docs/screenshot.png -->

## Docker (Phase 6)

```bash
docker compose up --build        # first build a few minutes (dependencies + model download)
# open http://localhost:8080      (API docs: http://localhost:8000/docs)
```

```
browser ──► web (nginx :8080) ──/api/*──► api (FastAPI :8000) ──► Groq / OpenAI
             static React build             embedding model baked into image
                                            ./data/raw  (PDFs, bind mount)
                                            chroma-data (vector index, named volume)
```

| Image | Highlights |
|---|---|
| `healthcare-navigator-api` | Multi-stage build, ONNX embeddings (no PyTorch), model baked in at build time (offline start), non-root user, liveness check |
| `healthcare-navigator-web` | Node build stage → nginx; `/api` reverse proxy (same origin, no CORS), SPA fallback, gzip, long-cache hashed assets, CSP and security headers |

- Put PDFs in `data/raw/`; on first start the API indexes them in the background (`AUTO_INGEST_ON_STARTUP`).
  Re-scan later with `docker compose exec api python -m app.ingestion` or `POST /api/ingest`.
- Secrets come from `.env` at runtime and are never baked into images.
- CI builds both images and smoke-tests the running stack on every push.

## Deployment (Phase 7)

**Live:** website on Vercel, API on Render's free tier, LLM on Groq: $0/month.
Step-by-step guide: [DEPLOY.md](DEPLOY.md).

```
Vercel (React) ──HTTPS──► Render (FastAPI in Docker, 512 MB) ──► Groq
                             └─ restores a prebuilt index from a GitHub Release at startup
```

- **Fits a 512 MB free instance:** PyTorch replaced by ONNX Runtime (FastEmbed); app libraries load in ~200 MB.
- **No persistent disk needed:** the index is exported once (`python -m app.ingestion --export`)
  and restored on cold start from `INDEX_URL`, with sha256 and embedding-model checks.
- **CI/CD:** GitHub Actions (backend, frontend, Docker end-to-end incl. real embeddings) →
  Render deploys only after checks pass; Vercel deploys every push with PR previews.
- **Cold starts handled in the UI:** "Waking up the server…" instead of an error.

<!-- Live demo: https://<project>.vercel.app · API docs: https://<service>.onrender.com/docs -->

## Roadmap

- [x] Phase 0 — Foundations (repo, config, logging, CI)
- [x] Phase 1 — Ingestion pipeline
- [x] Phase 2 — RAG chain + confidence scoring
- [x] Phase 3 — Evaluation
- [x] Phase 4 — REST API
- [x] Phase 5 — React frontend
- [x] Phase 6 — Docker
- [x] Phase 7 — Deployment + CI/CD
- [ ] Phase 8 — Hardening
