---
title: F-150 Spec Finder API
emoji: 🔧
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# F-150 Spec Finder — API

FastAPI service behind the Spec Finder UI. Retrieval runs on a prebuilt
index (`index/vectors.npz`), so this Space never needs the PDF.

Endpoints: `POST /api/extract`, `POST /api/search`, `GET /api/specs`,
`GET /api/meta`, `GET /api/health`. The UI is also served at `/`.

Secrets (Settings → Variables and secrets):

- `GROQ_API_KEY` — required for LLM mode; without it the API answers in rules mode
- `SPECRAG_CORS_ORIGINS` — your Vercel URL, e.g. `https://f150-spec-finder.vercel.app`
