"""Evidence-bearing inferences and recommendations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    code: str
    statement: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticContext:
    role: str
    confidence: float
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    operation: str
    disposition: str
    rationale: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Warning:
    code: str
    severity: str
    title: str
    message: str
    evidence: tuple[Evidence, ...] = ()
    implication: str = ""
    recommendation: str | None = None


@dataclass(frozen=True)
class KeyCandidate:
    columns: tuple[str, ...]
    kind: str
    confidence: float
    uniqueness_rate: float
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class GrainContext:
    status: str
    label: str | None
    columns: tuple[str, ...]
    confidence: float
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class RelationshipContext:
    source_column: str
    target_column: str
    kind: str
    coverage: float
    confidence: float
    evidence: tuple[Evidence, ...] = ()
