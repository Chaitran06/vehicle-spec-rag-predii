"""Ingest: PDF -> pages -> chunks + parsed spec rows -> embedded index.

Every intermediate is written to build/ as JSON, so each stage can be read
and debugged on its own — open build/chunks.json and you see exactly what the
retriever sees.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from specrag.chunk import chunk_pages
from specrag.clean import clean_pages
from specrag.config import CHUNKS_FILE, PAGES_FILE, TABLES_FILE
from specrag.index import recreate_collection, save_chunks, upsert_chunks
from specrag.pdf_text import read_pdf
from specrag.spec_tables import parse_pages


def ingest(pdf_path: Path, embed: bool = True, progress=None) -> dict:
    stats: dict = {}

    t = time.perf_counter()
    raw = read_pdf(pdf_path)
    pages = clean_pages(raw)
    PAGES_FILE.write_text(json.dumps([asdict(p) for p in pages], ensure_ascii=False), encoding="utf-8")
    stats["pages"] = len(pages)
    stats["ocr_pages"] = sum(p.ocr for p in pages)
    stats["extract_s"] = round(time.perf_counter() - t, 1)

    chunks = chunk_pages(pages)
    save_chunks(chunks, CHUNKS_FILE)
    stats["chunks"] = len(chunks)
    stats["documents"] = len({c.doc_id for c in chunks})
    stats["chunk_kinds"] = {k: sum(c.kind == k for c in chunks) for k in ("spec_table", "table", "procedure")}

    rows = parse_pages(pages)
    TABLES_FILE.write_text(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    stats["spec_rows"] = len(rows)

    if embed:
        t = time.perf_counter()
        recreate_collection()
        upsert_chunks(chunks, progress=progress)
        stats["embed_s"] = round(time.perf_counter() - t, 1)
    return stats
