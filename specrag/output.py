"""ExtractionResult -> JSON / CSV, in the shape the assignment asks for."""

import csv
import io
import json

from specrag.models import CSV_COLUMNS, ExtractionResult


def to_json(results: list[ExtractionResult], full: bool = False) -> str:
    """Default: the flat list of spec records, as in the assignment example.
    full=True: the whole result, including sources and rejected items."""
    if full:
        return json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False)
    records = []
    for r in results:
        for spec in r.specs:
            records.append({"query": r.query, **spec.model_dump(exclude={"chunk_id"})})
    return json.dumps(records, indent=2, ensure_ascii=False)


def to_csv(results: list[ExtractionResult]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["query", *CSV_COLUMNS], extrasaction="ignore")
    writer.writeheader()
    for r in results:
        for spec in r.specs:
            writer.writerow({"query": r.query, **spec.model_dump()})
    return buffer.getvalue()
