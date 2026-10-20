"""FastAPI wrapper around the pipeline, plus the web UI.

Locally, `specrag serve` runs this and serves frontend/ at /, so one command
gives you the whole app. Deployed, the same app runs on Hugging Face Spaces
(see deploy/) and the frontend is hosted on Vercel, calling it over CORS.
"""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from specrag import config

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and the index before accepting traffic, so the
    first visitor does not wait for it."""
    from specrag.embed import get_embedder
    from specrag.retrieve import get_store

    print("Loading embedding model and index...", flush=True)
    embedder = get_embedder()
    store = get_store()
    print(f"Ready: {embedder.name} ({embedder.dim}-dim), {len(store.chunks)} chunks", flush=True)
    yield


app = FastAPI(title="F-150 Spec Finder API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=config.MAX_QUERY_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=15)
    system: str | None = Field(default=None, description="Suspension | Driveline | Brakes")
    mode: str | None = Field(default=None, description="llm | rules")


# One process on a free tier: a dict of deques is enough. Behind several
# workers this becomes per-worker; move it to Redis before that matters.
_HITS: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]
    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= config.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests: the limit is {config.RATE_LIMIT_REQUESTS} per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS} seconds. Wait a moment and try again.",
        )
    window.append(now)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    """What the UI shows in its header and filters. Cheap: no model calls."""
    from specrag.extractor import load_rows
    from specrag.retrieve import get_store

    store = get_store()
    sections = {}
    for c in store.chunks:
        sections[c.section_id] = {"id": c.section_id, "title": c.section_title, "system": c.system}
    return {
        "chunks": len(store.chunks),
        "pages": max(c.page_end for c in store.chunks),
        "spec_rows": len(load_rows()),
        "sections": sorted(sections.values(), key=lambda s: s["id"]),
        "llm_available": bool(config.GROQ_API_KEY),
        "default_mode": config.DEFAULT_MODE,
        "llm_model": config.LLM_MODEL,
        "embed_model": Path(config.EMBED_MODEL).name,
    }


@app.post("/api/extract")
def extract(payload: ExtractRequest, request: Request):
    from specrag.extractor import extract as run

    mode = (payload.mode or config.DEFAULT_MODE).lower()
    if mode not in ("llm", "rules"):
        raise HTTPException(status_code=422, detail="mode must be 'llm' or 'rules'.")

    note = None
    if mode == "llm" and not config.GROQ_API_KEY:
        mode, note = "rules", "No LLM key is configured on this server, so the rule-based extractor answered."

    if mode == "llm":
        _rate_limit(request)

    try:
        result = run(payload.query, top_k=payload.top_k, system=payload.system or None, mode=mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # provider errors, imported lazily
        name = type(exc).__name__
        if name == "RateLimitError":
            raise HTTPException(
                status_code=429,
                detail="The language model's usage quota is used up for now. "
                "Switch to Rules mode, or try again in a minute.",
            )
        if name in ("APIStatusError", "APIConnectionError", "InternalServerError", "BadRequestError"):
            raise HTTPException(status_code=502, detail=f"The language model request failed ({name}).")
        raise

    if note:
        result.note = note if not result.note else f"{note} {result.note}"
    return result


@app.post("/api/search")
def search(payload: ExtractRequest):
    """Retrieval only — what the LLM would read for this query."""
    from specrag.retrieve import search as run

    return {"query": payload.query, "results": [h.to_dict() for h in run(payload.query, top_k=payload.top_k,
                                                                          system=payload.system or None)]}


@app.get("/api/specs")
def specs(system: str | None = None, spec_type: str | None = None, q: str | None = None, limit: int = 500):
    """Every spec row the deterministic parser found — the 'Browse' tab."""
    from dataclasses import asdict

    from specrag.clean import system_for
    from specrag.extractor import load_rows

    rows = []
    needle = (q or "").lower().strip()
    for r in load_rows():
        if system and system_for(r.section_id) != system:
            continue
        if spec_type and r.spec_type != spec_type:
            continue
        if needle and needle not in r.component.lower():
            continue
        rows.append({**asdict(r), "system": system_for(r.section_id)})
    return {"count": len(rows), "rows": rows[:limit]}


# ------------------------------------------------------------------
# Static UI (local dev; on Vercel the same files are served by Vercel)
# ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
