# Deploying Healthcare Knowledge Navigator (free tier)

```
Vercel (React, global CDN) ──HTTPS──► Render free (FastAPI, Docker, 512 MB) ──► Groq LLM
                                         │ on cold start
                                         └─ downloads prebuilt index from a GitHub Release
```

| Piece | Host | Cost | Notes |
|---|---|---|---|
| Website | Vercel Hobby | free | Static build, auto-deploys on push |
| API | Render free web service | free | 0.1 CPU / 512 MB, sleeps after 15 min idle (~1 min wake-up), no persistent disk |
| Vector index | GitHub Release asset | free | Downloaded by the API at startup (`INDEX_URL`) |
| LLM | Groq | free tier | OpenAI-compatible API |

## 1. Build the index snapshot (on your machine, with Docker)

The embedding engine is now FastEmbed (ONNX), so rebuild the index once with it:

```bash
docker compose down -v            # -v deletes the old index volume (built with PyTorch)
docker compose up --build -d      # AUTO_INGEST rebuilds the index from data/raw
docker compose logs -f api        # wait for "Auto-ingest finished: ..." then Ctrl+C
docker compose exec api python -m app.ingestion --export /data/export/index.tar.gz
```

The last command prints the **file size and sha256**. The file is at `data/export/index.tar.gz`.

> Only publish documents you are allowed to redistribute (e.g. open-access guidelines).
> The snapshot contains passages of text from your PDFs.

## 2. Publish the snapshot as a GitHub Release asset

GitHub repo → **Releases** → **Draft a new release** → tag `index-v1` → attach
`data/export/index.tar.gz` → **Publish**. Copy the asset link; it looks like:

```
https://github.com/<you>/healthcare-navigator/releases/download/index-v1/index.tar.gz
```

The repository must be **public** for the API to download it without credentials.

## 3. Deploy the API on Render

1. https://render.com → sign in with GitHub.
2. **New → Blueprint** → choose the `healthcare-navigator` repo (it reads `render.yaml`).
3. Fill in the secret values it asks for:

| Variable | Value |
|---|---|
| `OPENAI_API_KEY` | your Groq key (`gsk_...`) |
| `INDEX_URL` | the release asset link from step 2 |
| `INDEX_SHA256` | the sha256 printed in step 1 |
| `CORS_ORIGINS` | leave as `http://localhost:5173` for now; set after step 4 |
| `CORS_ORIGIN_REGEX` | leave empty for now |

4. **Apply**. The first build takes ~5–10 min. Then open
   `https://<service>.onrender.com/health` → expect `"status":"ok"` and your document count.

## 4. Deploy the website on Vercel

1. https://vercel.com → sign in with GitHub → **Add New → Project** → import the repo.
2. **Root Directory:** `frontend` (framework is detected as Vite).
3. **Environment Variable:** `VITE_API_URL` = `https://<service>.onrender.com`
4. **Deploy** → you get `https://<project>.vercel.app`.

## 5. Connect them (CORS)

Render → your service → **Environment**:

- `CORS_ORIGINS` = `https://<project>.vercel.app`
- `CORS_ORIGIN_REGEX` = `https://<project>-.*\.vercel\.app` (allows Vercel preview URLs)

Save. Render redeploys automatically.

## 6. Verify

- `https://<service>.onrender.com/health/live` → `{"status":"alive"}`
- `https://<service>.onrender.com/docs` → API docs
- `https://<project>.vercel.app` → green "Online · N documents", ask a question, check citations
- After 15+ min idle, reload: the header shows "Waking up the server…" for up to ~1 min.

## Continuous deployment

- **CI** (GitHub Actions) runs backend tests, frontend tests and a Docker end-to-end test.
- **Render** redeploys the API on pushes to `main` that touch `backend/`, only after CI
  passes (`autoDeployTrigger: checksPass`).
- **Vercel** redeploys the site on every push and creates preview URLs for pull requests.
- **Updating documents:** re-run step 1, publish a new release (`index-v2`), update
  `INDEX_URL` and `INDEX_SHA256` on Render.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Site shows "Server unreachable" | Open `/health` on Render directly; check Render logs |
| Browser console: CORS error | `CORS_ORIGINS` must exactly match the Vercel URL (no trailing slash) |
| `/health` says "index is empty" | Check Render logs for "Index restore failed"; verify `INDEX_URL` opens in a browser and `INDEX_SHA256` matches |
| Render: "Out of memory" | Keep `EMBEDDING_THREADS=1`; lower `RETRIEVAL_TOP_K`; or move to a 1 GB plan |
| `llm_rate_limited` errors | Groq free-tier limit reached; wait, or switch `LLM_MODEL` |
