"""Evidence-bearing inferences and recommendations."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Evidence:
    code: str
    statement: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze(self.details))


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
