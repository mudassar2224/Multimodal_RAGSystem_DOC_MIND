from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Modality(StrEnum):
    DOCUMENT = "document"
    TEXT = "text"
    STRUCTURED = "structured"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass(slots=True)
class SourceRef:
    document_id: str
    filename: str
    kind: str = "file"
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    object_key: str | None = None

    def display(self) -> str:
        location = ""
        if self.page:
            location = f" — page {self.page}"
        elif self.slide:
            location = f" — slide {self.slide}"
        elif self.sheet:
            location = f" — Sheet {self.sheet}" + (f" — {self.cell_range}" if self.cell_range else "")
        elif self.start_seconds is not None:
            def clock(value: float) -> str:
                whole = int(value)
                return f"{whole // 3600:02}:{(whole % 3600) // 60:02}:{whole % 60:02}"
            location = f" — {clock(self.start_seconds)}–{clock(self.end_seconds or self.start_seconds)}"
        return f"{self.filename}{location}"


@dataclass(slots=True)
class ChunkDraft:
    text: str
    source: SourceRef
    visual_object_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Evidence:
    chunk_id: str
    text: str
    source: SourceRef
    score: float
    visual_object_key: str | None = None
