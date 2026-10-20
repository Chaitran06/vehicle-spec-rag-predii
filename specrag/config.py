"""All tunables live here, every one of them env-driven.

Nothing else in the package reads os.getenv directly. Same rule as ytrag.
"""

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# huggingface_hub warns on every model load that requests are unauthenticated.
# Advice about download rate limits, not a problem.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

load_dotenv(find_dotenv(usecwd=True))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Input + runtime data
# ------------------------------------------------------------------
# The manual is NOT committed (43 MB, and it is Ford's copyrighted material).
# Drop it in data/ or point SPECRAG_PDF at it.
PDF_PATH = Path(os.getenv("SPECRAG_PDF", PROJECT_ROOT / "data" / "sample-service-manual.pdf"))

# Everything derived from the PDF lives here and is cheap to rebuild:
#   pages.json   cleaned text per page      (seconds)
#   chunks.json  structure-aware chunks     (instant)
#   tables.json  parsed spec rows           (instant)
#   qdrant/      embedded local vector store (~1 min on CPU)
DATA_DIR = Path(os.getenv("SPECRAG_DATA_DIR", PROJECT_ROOT / "build"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAGES_FILE = DATA_DIR / "pages.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
TABLES_FILE = DATA_DIR / "tables.json"

# The deploy bundle: vectors + chunk payloads in one .npz, so the hosted API
# never needs the PDF or a re-embed at startup (same trick as ytrag).
INDEX_EXPORT = Path(os.getenv("SPECRAG_INDEX_EXPORT", PROJECT_ROOT / "index" / "vectors.npz"))


# ------------------------------------------------------------------
# PDF extraction
# ------------------------------------------------------------------
# Words whose vertical centres are within this many points are the same
# visual row. 3pt is about a third of the body font height in this manual —
# large enough to merge "Nm" and "lb-ft" cells that sit a hair lower than the
# description, small enough to never merge two real lines.
ROW_TOLERANCE = float(os.getenv("SPECRAG_ROW_TOLERANCE", 3.0))
# A horizontal gap wider than this between two words is a column break and is
# written as two spaces. The table parser relies on that double space.
COLUMN_GAP = float(os.getenv("SPECRAG_COLUMN_GAP", 12.0))
# Pages with less extractable text than this are OCR candidates. This manual
# has a text layer on all 852 pages, so OCR never fires on it — it exists for
# scanned manuals. Requires the Tesseract binary; skipped cleanly if missing.
OCR_MIN_CHARS = int(os.getenv("SPECRAG_OCR_MIN_CHARS", 50))
OCR_ENABLED = os.getenv("SPECRAG_OCR", "auto")  # auto | off


# ------------------------------------------------------------------
# Chunking
# ------------------------------------------------------------------
# ~900 chars is one procedure step group or one small table. Spec tables are
# never split regardless of this (a half table is a wrong answer waiting to
# happen), so this only governs prose.
CHUNK_CHARS = int(os.getenv("SPECRAG_CHUNK_CHARS", 900))
CHUNK_OVERLAP_LINES = int(os.getenv("SPECRAG_CHUNK_OVERLAP_LINES", 2))
MIN_CHUNK_CHARS = int(os.getenv("SPECRAG_MIN_CHUNK_CHARS", 60))


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------
# all-MiniLM-L6-v2 again. Same reasoning as the DSA project: small, fast on
# CPU, 384-dim. On this corpus exact part names matter more than semantic
# nuance, and BM25 covers that side (see index.py), so a bigger encoder buys
# little. Swap via env and run `specrag ingest` again.
EMBED_MODEL = os.getenv("SPECRAG_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH = int(os.getenv("SPECRAG_EMBED_BATCH", 32))
EMBED_QUERY_PREFIX = os.getenv("SPECRAG_EMBED_QUERY_PREFIX", "")


# ------------------------------------------------------------------
# Vector store (Qdrant, embedded local mode by default)
# ------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PATH = Path(os.getenv("SPECRAG_QDRANT_PATH", DATA_DIR / "qdrant"))
COLLECTION = os.getenv("SPECRAG_COLLECTION", "f150_manual")
UPSERT_BATCH = int(os.getenv("SPECRAG_UPSERT_BATCH", 128))


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
TOP_K = int(os.getenv("SPECRAG_TOP_K", 6))
# Each retriever over-fetches this many candidates before fusion.
CANDIDATES = int(os.getenv("SPECRAG_CANDIDATES", 30))
# Reciprocal Rank Fusion constant. 60 is the value from the original RRF
# paper (Cormack et al., 2009) and is what most hybrid-search systems ship.
RRF_K = int(os.getenv("SPECRAG_RRF_K", 60))
# Spec-table chunks get a small fusion bonus when the query asks for a spec.
# The same bolt shows up in a Removal procedure ("Tighten to 35 Nm") and in
# the Specifications table; the table is the authoritative, complete source.
# Measured on the golden set: 0.001 recovers one query over no boost; 0.002
# and 0.004 change nothing further. Bigger is worse in a way the eval does not
# show — it buries the procedure chunks that "Refer to the procedure" rows
# point at — so the smallest value that helps wins.
TABLE_BOOST = float(os.getenv("SPECRAG_TABLE_BOOST", 0.001))


# ------------------------------------------------------------------
# LLM extraction
# ------------------------------------------------------------------
# The assignment suggests Llama-3. Groq retired llama-3.3-70b-versatile and
# llama-3.1-8b-instant on 2026-08-16 and points at gpt-oss as the replacement,
# so that is the default. Any Groq chat model works via env.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("SPECRAG_LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("SPECRAG_LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("SPECRAG_LLM_MAX_TOKENS", 2048))
# How long one call may spend waiting out Groq rate limits before giving up.
# 120 s suits the CLI and the eval. For the public API set it lower (e.g. 20)
# so a visitor gets the "quota used up" message instead of a long spinner.
LLM_RATE_LIMIT_WAIT = float(os.getenv("SPECRAG_LLM_RATE_LIMIT_WAIT", 120))
# "llm"   -> retrieval + LLM extraction + grounding check (the real pipeline)
# "rules" -> retrieval + deterministic table parser, no LLM. Baseline for the
#            eval, and the fallback when no key is configured or quota is gone.
DEFAULT_MODE = os.getenv("SPECRAG_MODE", "llm" if GROQ_API_KEY else "rules")


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
MAX_QUERY_CHARS = int(os.getenv("SPECRAG_MAX_QUERY_CHARS", 300))
RATE_LIMIT_REQUESTS = int(os.getenv("SPECRAG_RATE_LIMIT_REQUESTS", 20))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("SPECRAG_RATE_LIMIT_WINDOW", 60))
# Comma-separated origins allowed to call the API from a browser — your
# Vercel URL goes here. "*" is fine for a demo.
CORS_ORIGINS = [o.strip() for o in os.getenv("SPECRAG_CORS_ORIGINS", "*").split(",") if o.strip()]
