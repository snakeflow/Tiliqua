"""
Data abstractions for the Tiliqua document mining framework.

All models support as_dict() for backward compatibility with dict-based code.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum


class TaskType(str, Enum):
    RIS_METADATA = "RIS_METADATA"
    STATEMENT_EXTRACTION = "STATEMENT_EXTRACTION"
    REFERENCE_EXTRACTION = "REFERENCE_EXTRACTION"
    RELEVANCE_CLASSIFICATION = "RELEVANCE_CLASSIFICATION"
    METHOD_EXTRACTION = "METHOD_EXTRACTION"


class StatementType(str, Enum):
    ORIGINAL_CONTRIBUTION = "ORIGINAL_CONTRIBUTION"
    REFERENCED_STATEMENT = "REFERENCED_STATEMENT"
    DATA_METHOD = "DATA_METHOD"


@dataclass
class ExecutionConfig:
    model: str = "gpt-5-mini"
    tools: Optional[list] = None
    tool_choice: Optional[str] = None

    def as_dict(self) -> dict:
        return {"model": self.model, "tools": self.tools, "tool_choice": self.tool_choice}

    def get(self, key: str, default=None):
        """dict-compatible .get() for backward compatibility."""
        return getattr(self, key, default)


@dataclass
class Segment:
    doc_id: str
    segment_id: int
    text: str
    page: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    metadata: dict = field(default_factory=dict)
    raw_text: str = ""
    segments: list = field(default_factory=list)
    knowledge: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "FID": self.doc_id,
            "metadata": self.metadata,
            "raw_text": self.raw_text,
            "segments": [s.as_dict() if hasattr(s, "as_dict") else s for s in self.segments],
            "knowledge": self.knowledge,
        }


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    doc_id: str
    segment_id: Optional[int] = None
    prompt: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d


@dataclass
class Statement:
    doc_id: str
    segment_id: int
    type: StatementType
    text: str
    citation: Optional[str] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class RISRecord:
    doc_id: str
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CitationEdge:
    source_doc: str
    target_reference: str
    statement_id: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)
