"""Query -> retrieved chunks -> structured, grounded spec records.

The rule this module is built around: a torque value the model made up is
worse than no answer. Someone tightens a caliper bolt to it. So the model is
only trusted with the part it is good at — reading a messy excerpt and
deciding which row answers the question — and everything else is checked or
attached by code:

  * page, section and procedure come from the chunk metadata, never from the
    model, so a citation cannot be hallucinated;
  * every value must be found verbatim in the excerpt the model cited (or,
    if it cited the wrong number, in another retrieved excerpt) — otherwise it
    is moved to `rejected` and shown as such, not silently dropped;
  * the line it was found on becomes `evidence`, so the reader can check it
    in one glance.
"""

import json
import re
import time

from pydantic import ValidationError

from specrag.config import DEFAULT_MODE, LLM_MODEL, TABLES_FILE, TOP_K
from specrag.index import tokenize
from specrag.models import ExtractionResult, LLMSpec, SpecRecord, SpecRow
from specrag.retrieve import Hit, detect_spec_type, expand_query, search

SYSTEM_PROMPT = """You extract vehicle specifications from excerpts of the 2014 Ford F-150
Workshop Manual. You return JSON only.

Return: {"specs": [{"component": str, "spec_type": str, "value": str, "unit": str,
"alt_value": str|null, "alt_unit": str|null, "condition": str|null, "source": int}]}

Rules:
1. Use ONLY the excerpts. Never use your own knowledge of vehicles. If the
   excerpts do not contain the answer, return {"specs": []}.
2. Return only specs that answer the query, but ALL of them. A torque table
   has many rows: skip rows for unrelated parts, keep every row for the part
   asked about. A query naming a group of parts covers every member of it —
   "caliper bolts" means every brake caliper bolt row (anchor plate, flow,
   guide pin, support bracket ...). If the query is broad ("front suspension
   torque specs"), return every row of the matching table.
3. The same component often appears in several excerpts with DIFFERENT values
   (front vs rear brakes, RWD vs 4WD, SVT Raptor, 8.8-inch vs 9.75-inch axle).
   Return each as a separate item. Never merge or average.
   "condition" is a SHORT variant label of 1-4 words that tells these apart:
   "Front", "Rear", "RWD", "4WD", "SVT Raptor", "9.75-inch axle", "ELD only".
   Take it from the section title or the row text. Do not copy the section
   number or the full section title — they are reported separately. Use null
   when there is no variant.
4. Copy numbers exactly as printed, including ranges ("0.203-0.304") and
   thousands separators ("1,420"). Do not convert units.
5. Torque: value/unit in Nm; put the printed lb-ft or lb-in figure in
   alt_value/alt_unit. In torque tables the unit is in the column header
   ("Description  Nm lb-ft lb-in"); a "—" means no value in that unit.
6. component = the manual's own wording for the part, e.g.
   "Brake caliper guide pin bolts".
7. spec_type is one of: Torque, Fluid Capacity, Part Number, Material
   Specification, Runout, Backlash, Thickness, Preload, Pressure, Angle,
   Dimension. For Part Number, unit is "".
8. Skip rows that say "Refer to the procedure" instead of giving a value.
9. source = the [number] of the excerpt the value was read from."""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f"[{i}] {c.context_header} (pages {c.pages})\n{c.text}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# Grounding
# ------------------------------------------------------------------
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_DASHES)).strip()


def _value_pattern(value: str) -> re.Pattern:
    """Match the value as a whole token: '35' must not match '135' or '3.5'."""
    v = _norm(value).replace(",", "")
    v = re.escape(v).replace(r"\-", r"\s?-\s?")
    # After the value: a non-word character, the end, or a unit glued straight
    # on ("1.66L", "32mm") — the manual prints some capacities that way.
    after = r"(?=[^\w.]|\.(?!\d)|$|(?:L|ml|mm|in|pt|qt|oz|Nm|kPa|psi)\b)"
    # Before it: not a letter, digit, dot or hyphen, so the "1L" tail of a part
    # code like "XY-80W90-1L" is never mistaken for a 1 L capacity.
    return re.compile(rf"(?<![\w.-]){v}{after}")


def find_evidence(value: str, component: str, text: str) -> str | None:
    """The line in `text` containing `value`, preferring lines that also
    mention the component. None if the value is not in the text at all."""
    pattern = _value_pattern(value)
    comp_tokens = set(tokenize(component))
    best, best_overlap = None, -1
    for line in text.splitlines():
        if pattern.search(_norm(line).replace(",", "")):
            overlap = len(comp_tokens & set(tokenize(line)))
            if overlap > best_overlap:
                best, best_overlap = line.strip(), overlap
    return best


def _record(spec: dict, hit: Hit, evidence: str, verified: bool) -> SpecRecord:
    c = hit.chunk
    return SpecRecord(
        component=spec["component"].strip(),
        spec_type=spec["spec_type"].strip(),
        value=str(spec["value"]).strip(),
        unit=(spec.get("unit") or "").strip(),
        alt_value=(str(spec["alt_value"]).strip() if spec.get("alt_value") not in (None, "", "—") else None),
        alt_unit=(spec.get("alt_unit") or None),
        condition=spec.get("condition") or None,
        section=c.section_id,
        section_title=c.section_title,
        procedure=c.doc_title,
        pages=c.pages,
        chunk_id=c.chunk_id,
        evidence=evidence,
        verified=verified,
    )


_VALUE_WITH_UNIT = re.compile(r"\s*(\d[\d.,]*(?:\s?-\s?\d[\d.,]*)?)\s*([A-Za-z°][A-Za-z-]*)\s*")


def split_value_unit(value, unit) -> tuple[str, str]:
    """'1.66L' -> ('1.66', 'L'); '32mm' -> ('32', 'mm').

    The manual prints some capacities with no space ("1.66L"), and the model
    copies them verbatim into `value` — exactly as rule 4 of the prompt asks.
    Found in the first real eval run: 5 correct answers scored as wrong because
    the value carried its unit. Normalising here fixes the output for every
    consumer (JSON, CSV, UI), not just the eval.
    """
    value = "" if value is None else str(value).strip()
    unit = (unit or "").strip()
    m = _VALUE_WITH_UNIT.fullmatch(value)
    if m and (not unit or unit.lower() == m.group(2).lower()):
        return m.group(1), unit or m.group(2)
    return value, unit


def ground(specs: list[dict], hits: list[Hit]) -> tuple[list[SpecRecord], list[dict]]:
    """Keep specs whose value is in the cited excerpt; repair a wrong citation
    if the value is in a different excerpt; reject the rest."""
    kept: list[SpecRecord] = []
    rejected: list[dict] = []
    seen: set[tuple] = set()

    for spec in specs:
        spec = dict(spec)
        spec["value"], spec["unit"] = split_value_unit(spec.get("value"), spec.get("unit"))
        if spec.get("alt_value"):
            spec["alt_value"], spec["alt_unit"] = split_value_unit(spec["alt_value"], spec.get("alt_unit"))
        value = spec["value"]
        if not value or value in ("—", "-"):
            rejected.append({**spec, "reason": "empty value"})
            continue

        cited = spec.get("source")
        order = list(range(len(hits)))
        if isinstance(cited, int) and 1 <= cited <= len(hits):
            order.remove(cited - 1)
            order.insert(0, cited - 1)  # check the cited excerpt first

        record = None
        for idx in order:
            evidence = find_evidence(value, spec.get("component", ""), hits[idx].chunk.text)
            if evidence:
                record = _record(spec, hits[idx], evidence, verified=True)
                break

        if record is None:
            rejected.append({**spec, "reason": "value not found in any retrieved excerpt"})
            continue

        key = (record.component.lower(), record.value, record.unit, record.section, record.condition)
        if key not in seen:
            seen.add(key)
            kept.append(record)

    return kept, rejected


# ------------------------------------------------------------------
# Rules mode (no LLM)
# ------------------------------------------------------------------
_ROWS: list[SpecRow] | None = None


def load_rows() -> list[SpecRow]:
    global _ROWS
    if _ROWS is None:
        from specrag.config import PROJECT_ROOT

        for path in (TABLES_FILE, PROJECT_ROOT / "index" / "spec_rows.json"):
            if path.exists():
                _ROWS = [SpecRow(**r) for r in json.loads(path.read_text(encoding="utf-8"))]
                break
        else:
            _ROWS = []
    return _ROWS


def extract_rules(query: str, hits: list[Hit]) -> tuple[list[SpecRecord], list[dict]]:
    """Parsed table rows that sit inside the retrieved chunks, ranked by how
    many query words their component name shares."""
    spec_type = detect_spec_type(query)
    generic = {"torque", "nm", "capacity", "fill", "part", "number", "front", "rear"}
    q_tokens = set(tokenize(expand_query(query))) - generic
    # One shared word ("nut") matches half the manual. Ask for two when the
    # query has two to give; an empty answer beats a confident wrong one.
    min_overlap = 2 if len(set(tokenize(query)) - generic) >= 2 else 1

    candidates = []
    for hit in hits:
        text = _norm(hit.chunk.text)
        for row in load_rows():
            if row.section_id != hit.chunk.section_id or not (hit.chunk.page_start <= row.page <= hit.chunk.page_end):
                continue
            if _norm(row.line) not in text:
                continue
            if spec_type and row.spec_type != spec_type:
                continue
            overlap = len(q_tokens & set(tokenize(row.component)))
            if overlap >= min_overlap:
                candidates.append((overlap, row, hit))

    if not candidates:
        return [], []
    best = max(o for o, _, _ in candidates)
    records, seen = [], set()
    for overlap, row, hit in sorted(candidates, key=lambda t: -t[0]):
        if overlap < best:
            continue
        alt_value, alt_unit = (row.alt_values[0].split(" ", 1) + [None])[:2] if row.alt_values else (None, None)
        spec = {
            "component": row.component, "spec_type": row.spec_type, "value": row.value,
            "unit": row.unit, "alt_value": alt_value, "alt_unit": alt_unit,
            "condition": None,
        }
        record = _record(spec, hit, row.line, verified=True)
        key = (record.component.lower(), record.value, record.section)
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records, []


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def extract(
    query: str,
    top_k: int = TOP_K,
    system: str | None = None,
    mode: str | None = None,
    model: str = LLM_MODEL,
) -> ExtractionResult:
    mode = (mode or DEFAULT_MODE).lower()
    query = query.strip()
    timings: dict = {}

    t0 = time.perf_counter()
    hits = search(query, top_k=top_k, system=system)
    timings["retrieve"] = round((time.perf_counter() - t0) * 1000)

    result = ExtractionResult(
        query=query,
        mode=mode,
        model=model if mode == "llm" else None,
        sources=[h.to_dict() for h in hits],
    )
    if not hits:
        result.note = "Nothing relevant was retrieved."
        result.timings_ms = timings
        return result

    t1 = time.perf_counter()
    if mode == "rules":
        specs, rejected = extract_rules(query, hits)
    else:
        spec_hint = detect_spec_type(query)
        user = (
            f"QUERY: {query}\n"
            + (f"The query is asking for: {spec_hint}\n" if spec_hint else "")
            + f"\nEXCERPTS\n{build_context(hits)}"
        )
        from specrag.llm import chat_json

        raw = chat_json(SYSTEM_PROMPT, user, model=model)
        valid, malformed = [], []
        for item in raw.get("specs", []) if isinstance(raw, dict) else []:
            try:
                valid.append(LLMSpec.model_validate(item).model_dump())
            except ValidationError as exc:
                malformed.append({**(item if isinstance(item, dict) else {"raw": item}),
                                  "reason": f"malformed: {exc.errors()[0]['msg']}"})
        specs, rejected = ground(valid, hits)
        rejected = malformed + rejected
    timings["extract"] = round((time.perf_counter() - t1) * 1000)

    result.specs = specs
    result.rejected = rejected
    result.timings_ms = timings
    if not specs:
        result.note = "The retrieved sections do not state this specification."
    return result
