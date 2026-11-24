# Frontend

React 19 + TypeScript + Vite 7 + Tailwind CSS 4 + TanStack Query.

```bash
npm ci            # install exact locked versions
npm run dev       # http://localhost:5173 (calls /api/* -> http://localhost:8000)
npm test          # Vitest + Testing Library
npm run build     # typecheck + production build -> dist/
```

Set `VITE_API_URL` (see `.env.example`) when the API runs on another origin in production.

| Component | Role |
|---|---|
| `QueryForm` | Question input (Enter to ask), passage count, example questions |
| `AnswerPanel` | Renders the answer safely (no raw HTML), clickable `[n]` markers |
| `CitationCard` | Source title, file, page, snippet, similarity bar |
| `ConfidenceBadge` | Green / amber / red badge with the reasons behind the level |
| `Sidebar` | Recent questions (localStorage) and indexed documents |
| `Header` | Live server status from `/health` |
