"""Canonical result returned by :func:`oribi.inspect`."""

from dataclasses import dataclass
from typing import Any

from .evidence import GrainContext, KeyCandidate, Recommendation, RelationshipContext, SemanticContext, Warning
from .profile import DatasetProfile, StatisticsContext


@dataclass(frozen=True)
class ColumnContext:
    name: str
    dtype: str
    logical_type: str
    statistics: StatisticsContext
    semantic: SemanticContext
    recommendations: tuple[Recommendation, ...] = ()
    warnings: tuple[Warning, ...] = ()


@dataclass(frozen=True)
class DatasetContext:
    profile: DatasetProfile
    columns: tuple[ColumnContext, ...]
    candidate_keys: tuple[KeyCandidate, ...] = ()
    grain: GrainContext | None = None
    relationships: tuple[RelationshipContext, ...] = ()
    warnings: tuple[Warning, ...] = ()
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        from ..serialization.dict import to_dict
        return to_dict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        from ..serialization.dict import to_json
        return to_json(self, indent=indent)

    def to_markdown(self) -> str:
        from ..serialization.markdown import to_markdown
        return to_markdown(self)

    def to_context(self, *, max_chars: int = 8000) -> str:
        from ..serialization.context import to_context
        return to_context(self, max_chars=max_chars)

    def summary(self) -> str:
        rows = self.profile.row_count
        cols = self.profile.column_count
        parts = [f"Dataset: {rows:,} rows × {cols:,} columns."]
        if self.profile.duplicate_row_count:
            parts.append(f"{self.profile.duplicate_row_count:,} duplicate rows observed.")
        if self.grain and self.grain.status == "estimated" and self.grain.label:
            parts.append(f"Likely grain: {self.grain.label} (confidence {self.grain.confidence:.2f}).")
        elif self.grain:
            parts.append("Likely grain could not be determined confidently.")
        if self.warnings:
            parts.append(f"{len(self.warnings)} analytical warning(s) detected.")
        return " ".join(parts)
