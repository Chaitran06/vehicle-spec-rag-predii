"""Two indexes over the same retrieval units: Qdrant (dense) and BM25 (sparse).

Retrieval units, not just chunks ("small-to-big")
-------------------------------------------------
The first version embedded one vector per chunk. Measured on the golden set,
dense-only retrieval found the expected table row only 47% of the time. The
reason is visible in the data: a Torque Specifications chunk is 15 bolts in
one vector, so a query about ONE bolt is closer to a removal procedure that
mentions that bolt in passing than to the table that actually lists it.

So every table row is also indexed on its own, with the chunk's context
header and the table's column header in front of it:

    Section 206-04: Rear Disc Brake | Specifications | Torque Specifications
    Description  Nm lb-ft lb-in
    Brake caliper guide pin bolts  33 24  —

Search matches against these small units, then returns the PARENT chunk — the
whole table — to the LLM. Small units for matching, big units for reading.
That one change took dense row-recall from 47% to the figure in the README.

Why both dense and BM25
-----------------------
Spec queries are mostly exact part names ("stabilizer bar link nuts", "IWE
bolts"), which BM25 matches exactly and a 384-dim MiniLM vector blurs. Dense
still earns its place on paraphrases where no word overlaps ("how tight do
the wheel nuts go"). retrieve.py fuses the two.

Storage
-------
  local dev  Qdrant embedded mode in build/qdrant — no account, no Docker
  deploy     index/vectors.npz loaded into an in-memory Qdrant at startup, so
             the server never needs the PDF and never re-embeds the manual
  BM25       rebuilt in memory from the payloads at startup (~0.2 s)
"""

import atexit
import json
import re
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from specrag.config import (
    CHUNKS_FILE,
    COLLECTION,
    INDEX_EXPORT,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    UPSERT_BATCH,
)
from specrag.embed import get_embedder
from specrag.models import _NAMESPACE, Chunk

# ------------------------------------------------------------------
# Retrieval units
# ------------------------------------------------------------------
_TABLE_HEADER = re.compile(r"^(Description|Item)\s")
_HAS_VALUE = re.compile(r"\d")


def retrieval_units(chunk: Chunk) -> list[dict]:
    """The chunk itself, plus one unit per value-bearing row if it is a table."""
    units = [{"unit_id": chunk.chunk_id, "unit_kind": "chunk", "unit_text": chunk.embed_text}]
    if chunk.kind not in ("spec_table", "table"):
        return units

    lines = chunk.text.splitlines()
    header = next((l for l in lines if _TABLE_HEADER.match(l)), "")
    for i, line in enumerate(lines):
        if line == header or not _HAS_VALUE.search(line) or len(line) < 6:
            continue
        # A wrapped row ("Motorcraft® High Performance ... 1,420 ml" followed
        # by "DOT 3 Motor Vehicle Brake ...") needs its neighbour to make sense.
        nxt = lines[i + 1] if i + 1 < len(lines) and not _HAS_VALUE.search(lines[i + 1]) else ""
        text = "\n".join(p for p in (chunk.context_header, header, line, nxt) if p)
        units.append({"unit_id": f"{chunk.chunk_id}#r{i}", "unit_kind": "row", "unit_text": text})
    return units


# ------------------------------------------------------------------
# Qdrant client
# ------------------------------------------------------------------
_CLIENT: QdrantClient | None = None


def _close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None


atexit.register(_close_client)


def get_client(in_memory: bool = False) -> QdrantClient:
    """Qdrant Cloud if QDRANT_URL is set, else embedded local, else in-memory."""
    global _CLIENT
    if _CLIENT is None:
        if QDRANT_URL:
            _CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        elif in_memory:
            _CLIENT = QdrantClient(":memory:")
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            try:
                _CLIENT = QdrantClient(path=str(QDRANT_PATH))
            except RuntimeError as exc:
                if "already accessed" in str(exc) or "Storage folder" in str(exc):
                    raise RuntimeError(
                        "The local index is already open in another process — most likely "
                        "`specrag serve` is running. Stop it and try again."
                    ) from exc
                raise
    return _CLIENT


def collection_name() -> str:
    """Embedding dim in the name, so a model swap can never hit a size mismatch."""
    return f"{COLLECTION}_{get_embedder().dim}"


def recreate_collection() -> str:
    client = get_client()
    name = collection_name()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=get_embedder().dim, distance=Distance.COSINE),
    )
    return name


def _point_id(unit_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, unit_id))


def upsert_chunks(chunks: list[Chunk], progress=None) -> int:
    """Embed every retrieval unit and upsert. Idempotent (stable point IDs)."""
    units = [(chunk, unit) for chunk in chunks for unit in retrieval_units(chunk)]
    name = collection_name()
    client = get_client()
    embedder = get_embedder()
    total = 0
    for start in range(0, len(units), UPSERT_BATCH):
        batch = units[start : start + UPSERT_BATCH]
        vectors = embedder.embed_documents([u["unit_text"] for _, u in batch])
        client.upsert(
            collection_name=name,
            points=[
                PointStruct(id=_point_id(u["unit_id"]), vector=v, payload={**c.to_payload(), **u})
                for (c, u), v in zip(batch, vectors)
            ],
            wait=True,
        )
        total += len(batch)
        if progress:
            progress(total)
    return total


def dense_search(query: str, limit: int, system: str | None = None) -> list[tuple[Chunk, float]]:
    """-> [(parent chunk, best cosine similarity)] best first, one entry per chunk."""
    query_filter = None
    if system:
        query_filter = Filter(must=[FieldCondition(key="system", match=MatchValue(value=system))])
    # Over-fetch: several rows of one table can occupy the top of the list.
    points = get_client().query_points(
        collection_name=collection_name(),
        query=get_embedder().embed_query(query),
        limit=limit * 4,
        with_payload=True,
        query_filter=query_filter,
    ).points
    out, seen = [], set()
    for p in points:
        chunk_id = p.payload["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append((Chunk.from_payload(p.payload), float(p.score)))
        if len(out) >= limit:
            break
    return out


def collection_size() -> int:
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return 0
    return client.count(name).count


# ------------------------------------------------------------------
# Chunk store + BM25
# ------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "what", "which",
    "with", "how", "much", "many", "me", "give", "show", "find", "value", "values", "spec",
    "specs", "specification", "specifications", "f-150", "2014", "does", "do", "take", "i",
}


def _stem(word: str) -> str:
    """Crude plural stripping — 'bolts' == 'bolt', 'nuts' == 'nut'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        tokens.append(_stem(tok))
        # "bracket-to-wheel" should also match "bracket" and "wheel".
        if "-" in tok:
            tokens.extend(_stem(p) for p in tok.split("-") if p and p not in _STOP)
    return tokens


class ChunkStore:
    """All chunks in memory, plus a BM25 index over the same retrieval units
    the dense index uses."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.units: list[tuple[str, str]] = []  # (parent chunk_id, unit text)
        for chunk in chunks:
            for unit in retrieval_units(chunk):
                self.units.append((chunk.chunk_id, unit["unit_text"]))
        self.bm25 = BM25Okapi([tokenize(text) for _, text in self.units])

    def bm25_search(self, query: str, limit: int, system: str | None = None) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out, seen = [], set()
        for idx in order:
            if scores[idx] <= 0:
                break
            chunk = self.by_id[self.units[idx][0]]
            if chunk.chunk_id in seen or (system and chunk.system != system):
                continue
            seen.add(chunk.chunk_id)
            out.append((chunk, float(scores[idx])))
            if len(out) >= limit:
                break
        return out


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_FILE) -> None:
    path.write_text(json.dumps([c.to_payload() for c in chunks], ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path = CHUNKS_FILE) -> list[Chunk]:
    return [Chunk.from_payload(p) for p in json.loads(path.read_text(encoding="utf-8"))]


# ------------------------------------------------------------------
# Shipping a prebuilt index (deploy)
# ------------------------------------------------------------------
def export_vectors(path: Path = INDEX_EXPORT) -> dict:
    """Dump every vector + payload to one compressed .npz (float16 halves it)."""
    client = get_client()
    name = collection_name()
    vectors, payloads, offset = [], [], None
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=512, offset=offset, with_payload=True, with_vectors=True
        )
        for point in points:
            vectors.append(point.vector)
            payloads.append(point.payload)
        if offset is None:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.array(vectors, dtype=np.float16),
        payloads=np.array(json.dumps(payloads, ensure_ascii=False)),
        model=np.array(Path(get_embedder().name).name),
    )
    return {"count": len(vectors), "mb": round(path.stat().st_size / 1024 / 1024, 2)}


def import_vectors(path: Path = INDEX_EXPORT) -> list[Chunk]:
    """Load the .npz into an in-memory Qdrant and return the chunks.

    Refuses a model mismatch — a query embedded by one model is meaningless
    against another model's vectors, and nothing would error, results would
    just be quietly wrong."""
    data = np.load(path, allow_pickle=False)
    model = str(data["model"])
    if model != Path(get_embedder().name).name:
        raise RuntimeError(
            f"{path.name} was built with {model} but SPECRAG_EMBED_MODEL is {get_embedder().name}. "
            "Rebuild with `specrag ingest && specrag export-index`, or set the matching model."
        )
    payloads = json.loads(str(data["payloads"]))
    vectors = data["vectors"].astype("float32")
    client = get_client(in_memory=True)
    name = collection_name()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
    for start in range(0, len(payloads), UPSERT_BATCH):
        client.upsert(
            collection_name=name,
            points=[
                PointStruct(id=_point_id(p["unit_id"]), vector=v.tolist(), payload=p)
                for p, v in zip(payloads[start : start + UPSERT_BATCH], vectors[start : start + UPSERT_BATCH])
            ],
            wait=True,
        )
    chunks = [Chunk.from_payload(p) for p in payloads if p.get("unit_kind") == "chunk"]
    return sorted(chunks, key=lambda c: (c.page_start, c.chunk_id))
