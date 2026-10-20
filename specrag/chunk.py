"""Cleaned documents -> structure-aware chunks.

Deliberately not a generic text splitter. RecursiveCharacterTextSplitter would
happily cut a torque table after row 7 of 15, and then "Upper ball joint nut"
lands in a chunk whose header row ("Description  Nm lb-ft lb-in") is in a
different chunk. The model then has to guess which number is which unit.

So chunking follows the manual's own structure:

1. Each source document (one printed web page, 1-28 PDF pages) is chunked on
   its own. A chunk never spans two documents, so it never mixes sections —
   which matters because 204-01A (RWD) and 204-01B (4WD) contain the same bolt
   names with different values.
2. Inside a document, blocks start at headings: "Torque Specifications",
   "Removal", "Installation", ... plus any line directly above a table header
   row ("Description  Nm lb-ft lb-in", "Item  Specification").
3. Spec tables are atomic. One table, one chunk, however long.
4. Prose blocks are packed up to CHUNK_CHARS, preferring to cut before a
   numbered step, with a couple of lines of overlap.
"""

import re

from specrag.clean import group_documents, system_for
from specrag.config import CHUNK_CHARS, CHUNK_OVERLAP_LINES, MIN_CHUNK_CHARS
from specrag.models import Chunk, Page

# Headings that open a block even without a table under them.
_HEADINGS = {
    "removal", "installation", "disassembly", "assembly", "special tool(s)",
    "material", "part", "torque specifications", "general specifications",
    "inspection and verification", "visual inspection chart", "symptom chart",
    "principles of operation", "component location", "overview", "adjustment",
    "inspection", "cleaning", "checking", "activation", "removal and installation",
}
# Tables whose rows are specifications, and therefore must stay whole.
_SPEC_TABLES = ("torque specifications", "general specifications", "material", "specification", "capacit")
_TABLE_HEADER = re.compile(r"^(Description|Item)\s{2,}|^Item\s+(Specification|Number|Description)")
_STEP = re.compile(r"^\d+\.\s")
# Lines that can only be procedure prose, never a table row.
_PROSE_START = re.compile(r"^(\d+\.(\s|$)|NOTE:|NOTICE:|WARNING:|CAUTION:|•|All vehicles|Vehicles with)")
_PINPOINT = re.compile(r"^(Pinpoint Test [A-Z]+|Symptom Chart|PINPOINT TEST)", re.I)


def _is_heading(lines: list[str], i: int) -> bool:
    line = lines[i].strip()
    if not line or len(line) > 80:
        return False
    if line.lower() in _HEADINGS or _PINPOINT.match(line):
        return True
    # A short line right above a table header row is that table's title.
    nxt = lines[i + 1] if i + 1 < len(lines) else ""
    return bool(_TABLE_HEADER.match(nxt)) and not _TABLE_HEADER.match(line)


def _doc_title(doc: list[Page]) -> str:
    first = doc[0]
    if first.doc_type == "SPECIFICATIONS":
        return "Specifications"
    for line in first.lines[:3]:
        if line and not _TABLE_HEADER.match(line) and line.lower() not in _HEADINGS:
            return line.strip()
    return first.doc_type.title()


def _split_blocks(lines: list[str], pages: list[int]) -> list[dict]:
    """Cut the document at headings, and close tables where prose resumes.

    The second rule exists because a "Material" table at the top of a
    procedure has no closing heading — the numbered steps just start. Without
    it, the first version of this function produced a 6,375-character
    "table" that was really a whole brake inspection procedure.
    """
    blocks: list[dict] = []
    current = {"heading": "", "lines": [], "pages": []}
    in_table = False
    for i, line in enumerate(lines):
        heading = _is_heading(lines, i)
        prose_resumes = in_table and bool(_PROSE_START.match(line))
        if (heading or prose_resumes) and current["lines"]:
            blocks.append(current)
            current = {"heading": line.strip() if heading else "", "lines": [], "pages": []}
            in_table = False
        elif heading:
            current["heading"] = line.strip()
        if heading:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            in_table = bool(_TABLE_HEADER.match(nxt))
        current["lines"].append(line)
        current["pages"].append(pages[i])
    if current["lines"]:
        blocks.append(current)
    return blocks


def _is_spec_table(heading: str) -> bool:
    h = heading.lower()
    return any(key in h for key in _SPEC_TABLES)


def _pack(lines: list[str], pages: list[int]) -> list[tuple[list[str], list[int]]]:
    """Greedy pack prose lines into ~CHUNK_CHARS windows, cutting before steps."""
    windows = []
    start = 0
    while start < len(lines):
        size, end = 0, start
        while end < len(lines) and (size + len(lines[end]) < CHUNK_CHARS or end == start):
            size += len(lines[end]) + 1
            end += 1
        if end < len(lines):
            # Prefer to cut right before a numbered step in the back half.
            for k in range(end - 1, start + (end - start) // 2, -1):
                if _STEP.match(lines[k]):
                    end = k
                    break
        windows.append((lines[start:end], pages[start:end]))
        if end >= len(lines):
            break
        start = max(end - CHUNK_OVERLAP_LINES, start + 1)
    return windows


def chunk_document(doc: list[Page]) -> list[Chunk]:
    lines: list[str] = []
    line_pages: list[int] = []
    for page in doc:
        lines.extend(page.lines)
        line_pages.extend([page.page] * len(page.lines))
    if not lines:
        return []

    first = doc[0]
    title = _doc_title(doc)
    blocks = _split_blocks(lines, line_pages)

    # Fold tiny prose blocks (a lone "Special Tool(s)" list, say) into the
    # next prose block instead of embedding a 40-character chunk.
    merged: list[dict] = []
    for block in blocks:
        text_len = sum(len(l) for l in block["lines"])
        if (
            merged
            and not _is_spec_table(merged[-1]["heading"])
            and not _is_spec_table(block["heading"])
            and sum(len(l) for l in merged[-1]["lines"]) < MIN_CHUNK_CHARS
        ):
            merged[-1]["lines"] += block["lines"]
            merged[-1]["pages"] += block["pages"]
            merged[-1]["heading"] = merged[-1]["heading"] or block["heading"]
            continue
        if text_len == 0:
            continue
        merged.append(block)

    chunks: list[Chunk] = []
    for block in merged:
        heading = block["heading"]
        if _is_spec_table(heading):
            kind = "spec_table"
            windows = [(block["lines"], block["pages"])]
        else:
            is_table = len(block["lines"]) > 1 and bool(_TABLE_HEADER.match(block["lines"][1]))
            kind = "table" if is_table else "procedure"
            windows = _pack(block["lines"], block["pages"])

        for win_lines, win_pages in windows:
            text = "\n".join(win_lines).strip()
            if len(text) < 20:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{first.doc_id}@p{first.page}:{len(chunks)}",
                    doc_id=first.doc_id,
                    section_id=first.section_id,
                    section_title=first.section_title,
                    system=system_for(first.section_id),
                    doc_type=first.doc_type,
                    doc_title=title,
                    heading=heading,
                    kind=kind,
                    page_start=min(win_pages),
                    page_end=max(win_pages),
                    text=text,
                )
            )
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in group_documents(pages):
        chunks.extend(chunk_document(doc))
    return chunks
