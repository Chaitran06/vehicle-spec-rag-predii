"""Deterministic spec parser — no LLM.

Three jobs:

1. Ground truth. Every torque and general-spec table row in the manual is
   parsed here, and eval/golden.json is generated from those rows. That turns
   "the answers look right" into a measured number.
2. Baseline. `--mode rules` answers queries from these rows alone, so the LLM
   pipeline has something to beat in the eval.
3. Fallback. No GROQ_API_KEY, or the quota ran out: the UI still answers.

What it handles well: torque tables (one row per line, fixed unit columns),
"General Specifications" rows, inline "Tighten to 35 Nm (26 lb-ft)" steps and
part-number tables. What it handles badly, on purpose: the "Material" tables,
where the PDF interleaves three columns across four lines:

    Motorcraft® High Performance  WSS-M6C62-A 1,420 ml
    DOT 3 Motor Vehicle Brake  or WSS-M6C65- (3.0 pt)
    Fluid (US) / Motorcraft® Brake  A1

Capacities are pulled out best-effort; everything else in those tables is the
LLM's job. That split — rules where the layout is regular, the model where it
is not — is the honest one.
"""

import re

from specrag.clean import group_documents
from specrag.models import Page, SpecRow

_NUM = r"\d+(?:[.,]\d+)?"
_RANGE = rf"{_NUM}(?:\s?-\s?{_NUM})?"

# "Description  Nm lb-ft lb-in" / "Description  Nm lb-in"
_TORQUE_HEADER = re.compile(r"^Description\s{2,}((?:Nm|lb-ft|lb-in)(?:\s+(?:Nm|lb-ft|lb-in))*)\s*$")
# "Shock absorber lower nuts  90 66  —" / "Wheel nuts a  —  —"
_TORQUE_ROW = re.compile(rf"^(?P<desc>.*?[A-Za-z)].*?)(?:\s+(?P<foot>[a-e]))?\s+(?P<vals>(?:(?:{_NUM}|—)\s*){{1,3}})$")

_GENERAL_HEADER = re.compile(r"^Item\s{2,}Specification\b")
_SPEC_VALUE = re.compile(
    rf"(?P<value>{_RANGE})\s?(?P<unit>Nm|lb-ft|lb-in|mm|in|°|ml|L|qt|pt|oz|kPa|psi|bar)\b"
    rf"(?:\s*\((?P<alt>{_RANGE})\s?(?P<alt_unit>Nm|lb-ft|lb-in|mm|in|ml|L|qt|pt|oz|kPa|psi)\))?"
)
_ANGLE = re.compile(rf"(?P<value>{_NUM}°\s?-\s?{_NUM}°|{_NUM}°)")

_INLINE = re.compile(rf"Tighten to (?P<value>{_NUM}) Nm(?: \((?P<alt>{_NUM}) (?P<alt_unit>lb-ft|lb-in)\))?")
_VOLUME = re.compile(rf"(?<![-\w.,])(?P<value>{_NUM})\s?(?P<unit>ml|L|qt|pt|oz)\b(?:\s*\((?P<alt>{_NUM})\s?(?P<alt_unit>ml|L|qt|pt|oz)\))?")
_PART_HEADER = re.compile(r"^Item\s+Number\s+Description")
_PART_ROW = re.compile(r"^(?P<item>\d+)\s{2,}(?P<number>[A-Z0-9]{3,12}|—)\s{2,}(?P<desc>.+)$")
_MATERIAL_CODE = re.compile(r"^C?[A-Z]{1,3}-[0-9A-Z]")


def _spec_type_for(unit: str, description: str) -> str:
    d = description.lower()
    if unit in ("Nm", "lb-ft", "lb-in"):
        return "Preload" if "preload" in d else "Torque"
    if unit in ("ml", "L", "qt", "pt", "oz"):
        return "Fluid Capacity"
    if unit in ("kPa", "psi", "bar"):
        return "Pressure"
    if unit == "°":
        return "Angle"
    for word in ("runout", "backlash", "thickness", "clearance", "endplay", "wear", "taper", "diameter", "height"):
        if word in d:
            return word.title()
    return "Dimension"


def _row(page: Page, title: str, component, spec_type, value, unit, alt, source, line) -> SpecRow:
    return SpecRow(
        component=component.strip(" •"),
        spec_type=spec_type,
        value=value.replace(" ", ""),
        unit=unit,
        alt_values=alt,
        section_id=page.section_id,
        section_title=page.section_title,
        doc_title=title,
        page=page.page,
        source=source,
        line=line.strip(),
    )


def _doc_title(doc: list[Page]) -> str:
    # Local import avoids a cycle; chunk.py owns the title rule.
    from specrag.chunk import _doc_title as title_rule

    return title_rule(doc)


def parse_document(doc: list[Page]) -> list[SpecRow]:
    title = _doc_title(doc)
    rows: list[SpecRow] = []
    mode = None  # torque | general | material | part
    units: list[str] = []
    material_lines: list[tuple[Page, str]] = []

    flat = [(page, line) for page in doc for line in page.lines]
    for i, (page, line) in enumerate(flat):
        stripped = line.strip()

        # ---- table headers switch the parser mode ----
        if m := _TORQUE_HEADER.match(stripped):
            mode, units = "torque", m.group(1).split()
            continue
        if _GENERAL_HEADER.match(stripped):
            prev = flat[i - 1][1].strip().lower() if i else ""
            nxt = flat[i + 1][1] if i + 1 < len(flat) else ""
            is_material = prev == "material" or "Fill" in stripped or nxt.strip() == "Capacity"
            mode = "material" if is_material else "general"
            material_lines = []
            continue
        if _PART_HEADER.match(stripped):
            mode = "part"
            continue
        if re.match(r"^(\d+\.(\s|$)|NOTE:|NOTICE:|WARNING:|•|Removal$|Installation$)", stripped):
            mode = None

        # ---- inline instruction anywhere in a procedure ----
        for m in _INLINE.finditer(stripped):
            alt = [f"{m.group('alt')} {m.group('alt_unit')}"] if m.group("alt") else []
            component = title  # the procedure is the best component name we have
            rows.append(_row(page, title, component, "Torque", m.group("value"), "Nm", alt, "inline", stripped))

        if mode == "torque":
            m = _TORQUE_ROW.match(stripped)
            if not m:
                continue
            vals = m.group("vals").split()
            if len(vals) != len(units):
                continue
            by_unit = dict(zip(units, vals))
            primary_unit = "Nm" if "Nm" in by_unit and by_unit["Nm"] != "—" else None
            if primary_unit is None:
                continue  # "Wheel nuts a  —  —" -> refers to a procedure, no value
            alt = [f"{v} {u}" for u, v in by_unit.items() if u != "Nm" and v != "—"]
            rows.append(_row(page, title, m.group("desc"), "Torque", by_unit["Nm"], "Nm", alt, "table", stripped))

        elif mode == "general":
            parts = re.split(r"\s{2,}", stripped, maxsplit=1)
            if len(parts) != 2:
                continue
            desc, spec = parts
            if m := _SPEC_VALUE.search(spec):
                alt = [f"{m.group('alt')} {m.group('alt_unit')}"] if m.group("alt") else []
                rows.append(_row(page, title, desc, _spec_type_for(m.group("unit"), desc),
                                 m.group("value"), m.group("unit"), alt, "table", stripped))
            elif m := _ANGLE.search(spec):
                rows.append(_row(page, title, desc, "Angle", m.group("value").replace("°", ""), "°", [], "table", stripped))

        elif mode == "material":
            material_lines.append((page, stripped))
            m = _VOLUME.search(stripped)
            # Two false positives measured on this manual, both rejected here:
            #   "(3.0 pt)"     the lb/pt alternate of the line above, not a new row
            #   "1L (Canada)"  the tail of a wrapped part code like XY-80W90-1L,
            #                  which always sits in the first column
            first_col_end = len(re.split(r"\s{2,}", stripped)[0])
            if m and (m.start() < first_col_end or stripped[max(0, m.start() - 1)] == "("):
                m = None
            if m:
                # Component = first column of this line and the lines below it,
                # until the Motorcraft part code ("PM-1-C", "XL-3") starts.
                names = []
                for _, later in flat[i : i + 6]:
                    first_col = re.split(r"\s{2,}", later.strip())[0]
                    if names and (_MATERIAL_CODE.match(first_col) or first_col.startswith("Motorcraft")):
                        break
                    names.append(re.sub(rf"\s*{_NUM}\s?(ml|L|qt|pt|oz)\b.*$", "", first_col))
                component = " ".join(n for n in names if n).strip()
                alt_m = _VOLUME.search(flat[i + 1][1]) if i + 1 < len(flat) else None
                alt = [f"{m.group('alt')} {m.group('alt_unit')}"] if m.group("alt") else []
                if not alt and alt_m and "(" in flat[i + 1][1]:
                    alt = [f"{alt_m.group('value')} {alt_m.group('unit')}"]
                rows.append(_row(page, title, component, "Fluid Capacity", m.group("value"), m.group("unit"),
                                 alt, "table", stripped))

        elif mode == "part":
            if m := _PART_ROW.match(stripped):
                if m.group("number") != "—":
                    rows.append(_row(page, title, m.group("desc"), "Part Number", m.group("number"), "",
                                     [], "table", stripped))
    return rows


def parse_pages(pages: list[Page]) -> list[SpecRow]:
    rows: list[SpecRow] = []
    for doc in group_documents(pages):
        rows.extend(parse_document(doc))
    return rows
