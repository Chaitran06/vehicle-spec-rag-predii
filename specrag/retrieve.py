"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

RRF scores each chunk by 1 / (k + rank) in every list it appears in and sums.
It needs no score calibration — cosine similarities and BM25 scores live on
completely different scales, and averaging them directly lets whichever is
numerically larger win. Rank fusion sidesteps that.

On top of RRF, one domain nudge: if the query is asking for a spec (torque,
capacity, runout...) chunks that ARE spec tables get a small bonus. The same
bolt appears in a removal procedure ("Tighten to 35 Nm") and in the section's
Torque Specifications table; the table is the complete, authoritative one and
also lists the variants side by side.
"""

import re
from dataclasses import dataclass

from specrag.config import CANDIDATES, CHUNKS_FILE, INDEX_EXPORT, RRF_K, TABLE_BOOST, TOP_K
from specrag.index import ChunkStore, collection_size, dense_search, import_vectors, load_chunks, tokenize
from specrag.models import Chunk

# Query words -> the spec type being asked for. Used for the table boost and
# passed to the LLM as a hint. Order matters: first match wins.
SPEC_INTENTS = [
    ("Torque", r"\btorque|tighten|\bnm\b|lb-?ft|lb-?in|\bfoot.?pounds?"),
    ("Fluid Capacity", r"capacit|\bfill|\bfluid\b|lubricant|\boil\b|how much (oil|fluid)|litre|liter|quart"),
    ("Part Number", r"part (no|num)|part#|\bp/?n\b|part number|motorcraft"),
    ("Runout", r"runout|run-out"),
    ("Backlash", r"backlash"),
    ("Thickness", r"thickness|minimum thick|wear limit|discard"),
    ("Preload", r"preload"),
    ("Pressure", r"pressure|psi|kpa"),
]

# BM25 only sees the literal query, so a technician's word and the manual's
# word must be bridged here. Small and specific on purpose — every entry is a
# vocabulary mismatch seen while testing against this manual.
_EXPANSIONS = {
    r"\btorque\b": "torque specifications nm",
    r"\bcapacity\b": "fill capacity",
    r"\bbrake fluid\b": "dot 3 motor vehicle brake fluid",
    r"\bdiff(erential)? (oil|fluid)\b": "rear axle lubricant",
    r"\bgear oil\b": "axle lubricant",
    r"\blug nuts?\b": "wheel nuts",
    r"\bsway bar\b": "stabilizer bar",
    r"\bcaliper bracket\b": "caliper anchor plate support bracket",
    r"\brotor\b": "brake disc",
    r"\bcv (axle|shaft)\b": "halfshaft",
}


def detect_spec_type(query: str) -> str | None:
    q = query.lower()
    for name, pattern in SPEC_INTENTS:
        if re.search(pattern, q):
            return name
    return None


def expand_query(query: str) -> str:
    extra = [add for pattern, add in _EXPANSIONS.items() if re.search(pattern, query.lower())]
    return " ".join([query, *extra])


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused score, higher is better
    dense_rank: int | None
    bm25_rank: int | None

    def to_dict(self, preview_chars: int = 0) -> dict:
        c = self.chunk
        return {
            "chunk_id": c.chunk_id,
            "section": c.section_id,
            "section_title": c.section_title,
            "system": c.system,
            "procedure": c.doc_title,
            "heading": c.heading,
            "kind": c.kind,
            "pages": c.pages,
            "score": round(self.score, 5),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "text": c.text if not preview_chars else c.text[:preview_chars],
        }


# ------------------------------------------------------------------
# Store loading
# ------------------------------------------------------------------
_STORE: ChunkStore | None = None


def get_store() -> ChunkStore:
    """Local build (build/chunks.json + build/qdrant) if present, else the
    shipped index/vectors.npz loaded into memory. The second path is what the
    deployed API uses."""
    global _STORE
    if _STORE is None:
        if CHUNKS_FILE.exists() and collection_size() > 0:
            chunks = load_chunks()
        elif INDEX_EXPORT.exists():
            chunks = import_vectors(INDEX_EXPORT)
        else:
            raise RuntimeError(
                "No index found. Run `specrag ingest` (needs the PDF in data/), "
                "or ship index/vectors.npz."
            )
        _STORE = ChunkStore(chunks)
    return _STORE


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
def search(
    query: str,
    top_k: int = TOP_K,
    system: str | None = None,
    method: str = "hybrid",  # hybrid | dense | bm25 — the last two exist for the eval
) -> list[Hit]:
    store = get_store()
    spec_type = detect_spec_type(query)

    dense = dense_search(query, CANDIDATES, system) if method in ("hybrid", "dense") else []
    sparse = store.bm25_search(expand_query(query), CANDIDATES, system) if method in ("hybrid", "bm25") else []

    fused: dict[str, Hit] = {}
    for rank, (chunk, _) in enumerate(dense, start=1):
        hit = fused.setdefault(chunk.chunk_id, Hit(chunk, 0.0, None, None))
        hit.dense_rank = rank
        hit.score += 1.0 / (RRF_K + rank)
    for rank, (chunk, _) in enumerate(sparse, start=1):
        hit = fused.setdefault(chunk.chunk_id, Hit(chunk, 0.0, None, None))
        hit.bm25_rank = rank
        hit.score += 1.0 / (RRF_K + rank)

    if spec_type and method == "hybrid":
        for hit in fused.values():
            if hit.chunk.kind == "spec_table":
                hit.score += TABLE_BOOST

    ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    top = ranked[:top_k]
    if method == "hybrid":
        top += follow_references(query, top, store)
    return top


# "Wheel nuts a  —  —" + footnote "a Refer to the procedure in this section."
_REFER_ROW = re.compile(r"^(?P<desc>.+?)\s+[a-e](?:\s+—){1,3}\s*$")


def follow_references(query: str, hits: list[Hit], store) -> list[Hit]:
    """Some table rows hold no value, only "Refer to the procedure in this
    section". The torque for wheel nuts (204 Nm) exists ONLY inside the Wheel
    and Tire installation steps. A retriever that stops at the table returns
    a row with no number, and the LLM correctly says "not found".

    So: if a retrieved table has such a row for the component being asked
    about, fetch the best procedure chunk from the same section that mentions
    that component and a "Tighten to", and append it to the results — exactly
    what a technician does when they read the footnote.
    """
    q_tokens = set(tokenize(expand_query(query)))
    have = {h.chunk.chunk_id for h in hits}
    extra: list[Hit] = []
    for hit in hits:
        if hit.chunk.kind != "spec_table":
            continue
        for line in hit.chunk.text.splitlines():
            m = _REFER_ROW.match(line.strip())
            if not m:
                continue
            desc_tokens = set(tokenize(m.group("desc")))
            if len(q_tokens & desc_tokens) < max(1, len(desc_tokens) - 1):
                continue
            candidates = [
                (chunk, score)
                for chunk, score in store.bm25_search(f"{m.group('desc')} tighten to", 40)
                if chunk.section_id == hit.chunk.section_id
                and chunk.kind == "procedure"
                and "Tighten to" in chunk.text
                and chunk.chunk_id not in have
            ]
            if candidates:
                chunk = candidates[0][0]
                have.add(chunk.chunk_id)
                extra.append(Hit(chunk, 0.0, None, None))
    return extra
