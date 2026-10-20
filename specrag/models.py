"""The data structures the pipeline moves around.

Dataclasses for the internal stages (Page -> Chunk), Pydantic for anything
that crosses a boundary: what the LLM must return, and what the API returns.
"""

import uuid
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, ConfigDict, Field

# Fixed namespace so the same chunk_id always maps to the same Qdrant point
# ID — re-ingesting overwrites instead of duplicating (same as ytrag).
_NAMESPACE = uuid.UUID("2b1d5c9e-7f0a-4c53-9b8e-4f150a5e2014")


# ------------------------------------------------------------------
# Ingestion stages
# ------------------------------------------------------------------
@dataclass
class Page:
    """One PDF page after cleaning. Lines are visual rows, columns split by '  '."""

    page: int  # 1-based, as a human would cite it
    doc_id: str  # e.g. "SE241A01" — the source HTML file the page was printed from
    section_id: str  # e.g. "204-01A"
    section_title: str  # e.g. "Front Suspension — Rear Wheel Drive (RWD)"
    doc_type: str  # e.g. "SPECIFICATIONS", "REMOVAL AND INSTALLATION"
    lines: list[str] = field(default_factory=list)
    ocr: bool = False


@dataclass
class Chunk:
    """What actually gets embedded and retrieved."""

    chunk_id: str  # f"{doc_id}@p{first_page}:{n}" — stable across re-ingest, unique
    doc_id: str
    section_id: str
    section_title: str
    system: str  # Suspension | Driveline | Brakes
    doc_type: str
    doc_title: str  # e.g. "Lower Arm", "Specifications"
    heading: str  # nearest block heading, e.g. "Torque Specifications"
    kind: str  # spec_table | table | procedure | text
    page_start: int
    page_end: int
    text: str  # the body, as shown to the LLM and the user

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))

    @property
    def context_header(self) -> str:
        """Prepended before embedding and before the LLM sees the chunk.

        A row like "Shock absorber lower nuts  90 66 —" says nothing about
        which axle, which drivetrain, or even that it is a torque. The header
        puts that back. Same idea as prefixing the lecture title in ytrag.
        """
        parts = [f"Section {self.section_id}: {self.section_title}", self.doc_type.title(), self.doc_title]
        if self.heading and self.heading != self.doc_title:
            parts.append(self.heading)
        return " | ".join(p for p in parts if p)

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}"

    @property
    def pages(self) -> str:
        return str(self.page_start) if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"

    def to_payload(self) -> dict:
        """Qdrant payload. Flat, primitives only."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> "Chunk":
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__})


@dataclass
class SpecRow:
    """One row parsed deterministically out of a spec table or an inline
    "Tighten to ..." instruction. Ground truth for eval, and the no-LLM path."""

    component: str
    spec_type: str
    value: str
    unit: str
    alt_values: list[str]  # e.g. ["66 lb-ft"]
    section_id: str
    section_title: str
    doc_title: str
    page: int
    source: str  # "table" | "inline"
    line: str  # the raw line it came from


# ------------------------------------------------------------------
# LLM contract
# ------------------------------------------------------------------
class LLMSpec(BaseModel):
    """What the model is asked to return per spec. Deliberately small —
    everything the model does not need to decide (page, section) is attached
    afterwards from the chunk metadata, so it cannot be hallucinated."""

    # Models return 35 as often as "35"; accept both.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    component: str
    spec_type: str
    value: str
    unit: str
    alt_value: str | None = None
    alt_unit: str | None = None
    condition: str | None = None  # "SVT Raptor", "Rear", "4WD", "with 9.75-inch axle"
    source: int = Field(..., description="Excerpt number the value was read from")


class LLMResponse(BaseModel):
    specs: list[LLMSpec] = Field(default_factory=list)


# ------------------------------------------------------------------
# Output contract (CLI JSON/CSV + API)
# ------------------------------------------------------------------
class SpecRecord(BaseModel):
    # The four fields the assignment asks for, first and in that order.
    component: str
    spec_type: str
    value: str
    unit: str
    # Everything a technician needs to trust the number.
    alt_value: str | None = None
    alt_unit: str | None = None
    condition: str | None = None
    section: str
    section_title: str
    procedure: str
    pages: str
    chunk_id: str
    evidence: str  # the exact source line the value was found on
    verified: bool  # value + unit found verbatim in the cited excerpt


class ExtractionResult(BaseModel):
    query: str
    mode: str  # llm | rules
    model: str | None = None
    specs: list[SpecRecord] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)  # LLM output that failed grounding
    sources: list[dict] = Field(default_factory=list)  # retrieved chunks, for the UI
    timings_ms: dict = Field(default_factory=dict)
    note: str | None = None


# Output columns for CSV, in order.
CSV_COLUMNS = [
    "component", "spec_type", "value", "unit", "alt_value", "alt_unit",
    "condition", "section", "section_title", "procedure", "pages", "evidence", "verified",
]
