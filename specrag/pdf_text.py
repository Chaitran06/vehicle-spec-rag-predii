"""PDF -> visual rows of text, per page.

Why not just page.get_text()? Because on this manual it destroys the tables.
PyMuPDF's default output follows the PDF's internal text order, which for a
spec table is one cell per line:

    Shock absorber lower nuts
    90
    66
    —

and the reader can no longer tell which number belongs to which bolt, or
which number is Nm and which is lb-ft. That is the single most damaging thing
that can happen to a spec-extraction pipeline, and no amount of prompting
fixes it downstream.

So we take word coordinates instead and rebuild rows the way a person reads
them: group words whose vertical centres line up, sort left to right, and mark
wide horizontal gaps as column breaks with a double space:

    Shock absorber lower nuts  90 66  —
"""

import pymupdf

from specrag.config import COLUMN_GAP, OCR_ENABLED, OCR_MIN_CHARS, ROW_TOLERANCE


def page_rows(page: pymupdf.Page, textpage=None) -> list[str]:
    """Rebuild visual rows from word boxes. Columns are joined by two spaces."""
    # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    words = page.get_text("words", textpage=textpage)
    if not words:
        return []

    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))

    rows: list[tuple[float, list]] = []
    for w in words:
        y_centre = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - y_centre) <= ROW_TOLERANCE:
            rows[-1][1].append(w)
        else:
            rows.append((y_centre, [w]))

    lines = []
    for _, row in rows:
        row.sort(key=lambda w: w[0])
        text = row[0][4]
        for prev, cur in zip(row, row[1:]):
            gap = cur[0] - prev[2]
            text += ("  " if gap > COLUMN_GAP else " ") + cur[4]
        lines.append(text)
    return lines


def _ocr_textpage(page: pymupdf.Page):
    """OCR fallback for pages with no text layer (scanned manuals).

    Uses PyMuPDF's Tesseract bridge. Returns None if Tesseract is not
    installed, so a missing system binary degrades to "page skipped" rather
    than a crashed ingest.
    """
    try:
        return page.get_textpage_ocr(full=True, dpi=300)
    except Exception:
        return None


def read_pdf(path) -> list[dict]:
    """-> [{"page": 1-based int, "rows": [...], "ocr": bool}] for every page."""
    out = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            rows = page_rows(page)
            used_ocr = False

            if OCR_ENABLED != "off" and sum(len(r) for r in rows) < OCR_MIN_CHARS:
                textpage = _ocr_textpage(page)
                if textpage is not None:
                    rows = page_rows(page, textpage=textpage)
                    used_ocr = True

            out.append({"page": index + 1, "rows": rows, "ocr": used_ocr})
    return out
