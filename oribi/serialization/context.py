"""Compact AI-oriented rendering of an existing context snapshot."""

from datetime import date, datetime
from numbers import Number
from typing import Any


def _format_aggregations(operations: Any) -> str:
    """Render aggregation names without treating a single string as a sequence."""
    if isinstance(operations, str):
        names = [operations] if operations else []
    else:
        names = list(operations)
    return " \u00b7 ".join(names) if names else "\u2014"


def _format_range_value(value: Any) -> str:
    """Render observed bounds without redundant midnight or decimal suffixes."""
    if isinstance(value, datetime):
        if value.time().replace(tzinfo=None).isoformat() == "00:00:00":
            return value.date().isoformat()
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Number) and not isinstance(value, (bool, complex)):
        try:
            integer = int(value)
            if value == integer:
                return str(integer)
        except (OverflowError, TypeError, ValueError):
            pass
    return str(value)


def to_context(context: Any, *, max_chars: int = 8000) -> str:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    p = context.profile
    header = ["DATASET", f"{p.row_count:,} rows \u00d7 {p.column_count:,} columns"]
    if p.duplicate_row_count:
        header.append(f"Duplicate rows: {p.duplicate_row_count:,}")

    grain_lines = ["GRAIN"]
    if context.grain and context.grain.status == "estimated":
        grain_columns = " \u00b7 ".join(context.grain.columns) or context.grain.label or "Undetermined"
        grain_lines.append(f"{grain_columns} \u00b7 {context.grain.confidence:.0%} confidence")
    else:
        grain_lines.append("Undetermined")

    all_warnings = list(context.warnings)
    for col in context.columns:
        all_warnings.extend(col.warnings)
    warning_lines = ["WARNINGS"] if all_warnings else []
    seen_warnings: set[tuple[str, str]] = set()
    for warning in all_warnings:
        if warning.code == "high_missingness":
            column = next((c for c in context.columns if warning in c.warnings), None)
            if column is None:
                continue
            message = f"\u26a0 {column.name} \u00b7 {column.statistics.missing_rate:.0%} missing"
        else:
            message = f"\u26a0 {warning.message}"
        key = (warning.code, message)
        if key not in seen_warnings:
            warning_lines.append(message)
            seen_warnings.add(key)
    column_lines = ["COLUMNS"]
    ranked = sorted(context.columns, key=lambda c: c.name)
    for col in ranked:
        s = col.statistics
        aggregations = [
            rec.operation for rec in col.recommendations
            if rec.disposition in {"recommended", "caution"}
        ]
        fields = [
            "",
            col.name,
            f"{col.dtype} \u00b7 {col.semantic.role} \u00b7 {col.semantic.confidence:.0%} confidence",
        ]
        stats = f"Missing {s.missing_rate:.0%} \u00b7 Unique {s.distinct_count:,}"
        if s.minimum is not None or s.maximum is not None:
            stats += f" \u00b7 Range {_format_range_value(s.minimum)} \u2192 {_format_range_value(s.maximum)}"
        fields.append(stats)
        fields.append(f"Aggregations: {_format_aggregations(aggregations)}")
        column_lines.extend(fields)

    if context.relationships:
        column_lines.append("RELATIONSHIPS")
        column_lines.extend(
            f"{rel.source_column} \u2192 {rel.target_column}: {rel.kind}, coverage {rel.coverage:.1%}"
            for rel in context.relationships
        )
    sections = [header, grain_lines]
    if warning_lines:
        sections.append(warning_lines)
    sections.append(column_lines)
    lines = [line for index, section in enumerate(sections) for line in (([""] if index else []) + section)]
    result = "\n".join(lines)
    if len(result) <= max_chars:
        return result
    marker = "[Context truncated.]"
    selected: list[str] = []
    for line in lines:
        candidate = "\n".join((*selected, line, marker))
        if len(candidate) <= max_chars:
            selected.append(line)
    if not selected:
        return marker[:max_chars]
    return "\n".join((*selected, marker))
