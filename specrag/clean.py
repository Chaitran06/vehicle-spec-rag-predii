"""Raw rows -> cleaned Pages with document metadata.

This manual is 177 workshop-manual web pages printed to PDF, 852 pages in
all. Printing left the same noise on every page:

    2014 F-150 Workshop Manual  Page 2 sur 5                    <- browser header
    ...
    file:///C:/TSO/tsocache/.../file=SE241A07.HTM~gen... repair4less  2014-03-01   <- footer

Both are pure noise for retrieval — "Page 2 sur 5" matches every query about
page counts and the footer URL is identical on 852 pages — so they go. But the
footer is also the only thing on a continuation page that says which document
it belongs to (the SE241A07.HTM file name), so it is read before it is thrown
away. Only the first page of each document carries the SECTION and document
type lines; continuation pages inherit them via that file name.
"""

import re

from specrag.models import Page

_HEADER = re.compile(r"Page \d+ sur \d+")
_FOOTER = re.compile(r"^file:///")
_DOC_ID = re.compile(r"file=([A-Z0-9]+)\.HT")
_SECTION = re.compile(r"^SECTION (\d{3}-\d{2}[A-Z]?):\s*(.+?)(?:\s{2,}2014 F-150 Workshop Manual)?$")
_DOC_TYPE = re.compile(r"^([A-Z][A-Z &/-]+?)\s{2,}Procedure revision date:.*$")
# The manual's bullet glyph is a Wingdings character that extracts as "z".
_BULLET = re.compile(r"(^|(?<=\s))z(?=\s)")
_NOISE = re.compile(r"\s*\b(repair4less|2014 F-150 Workshop Manual)\b\s*")

SYSTEMS = {"204": "Suspension", "205": "Driveline", "206": "Brakes"}


def system_for(section_id: str) -> str:
    return SYSTEMS.get(section_id[:3], "Other")


def _clean_line(line: str) -> str:
    line = _BULLET.sub("•", line)
    line = _NOISE.sub("  ", line)
    return line.strip()


def clean_pages(raw_pages: list[dict]) -> list[Page]:
    """Strip header/footer noise and attach section + document metadata."""
    doc_meta: dict[str, dict] = {}
    pages: list[Page] = []

    for raw in raw_pages:
        rows = list(raw["rows"])

        if rows and _HEADER.search(rows[0]):
            rows = rows[1:]

        doc_id = "UNKNOWN"
        if rows and _FOOTER.match(rows[-1]):
            match = _DOC_ID.search(rows[-1])
            doc_id = match.group(1) if match else "UNKNOWN"
            rows = rows[:-1]

        # First page of a document: SECTION line, then document type line.
        if rows and (match := _SECTION.match(rows[0].strip())):
            meta = {"section_id": match.group(1), "section_title": match.group(2).strip()}
            rows = rows[1:]
            if rows and (dt := _DOC_TYPE.match(rows[0].strip())):
                meta["doc_type"] = dt.group(1).strip()
                rows = rows[1:]
            else:
                meta["doc_type"] = "GENERAL"
            doc_meta[doc_id] = meta

        meta = doc_meta.get(doc_id, {"section_id": "000-00", "section_title": "Unknown", "doc_type": "GENERAL"})
        lines = [cleaned for row in rows if (cleaned := _clean_line(row))]

        pages.append(
            Page(
                page=raw["page"],
                doc_id=doc_id,
                section_id=meta["section_id"],
                section_title=meta["section_title"],
                doc_type=meta["doc_type"],
                lines=lines,
                ocr=raw.get("ocr", False),
            )
        )
    return pages


def group_documents(pages: list[Page]) -> list[list[Page]]:
    """Consecutive pages with the same doc_id form one source document.

    Exact duplicates are dropped. This PDF contains "Shock Absorber and Spring
    Assembly — SVT Raptor" (SE241B12) twice, pages 63-72 and 84-93, line for
    line identical. Indexing both would fill two of the six retrieval slots
    with the same text and count every value in it twice.
    """
    docs: list[list[Page]] = []
    for page in pages:
        if docs and docs[-1][-1].doc_id == page.doc_id:
            docs[-1].append(page)
        else:
            docs.append([page])

    unique, seen = [], set()
    for doc in docs:
        key = (doc[0].doc_id, hash(tuple(line for p in doc for line in p.lines)))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique
