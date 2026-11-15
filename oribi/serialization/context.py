"""Compact AI-oriented rendering of an existing context snapshot."""

from typing import Any


def to_context(context: Any, *, max_chars: int = 8000) -> str:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    p = context.profile
    header = [f"DATASET (observed): {p.row_count} rows, {p.column_count} columns."]
    if p.duplicate_row_count:
        header.append(f"Duplicate rows (observed): {p.duplicate_row_count}.")
    grain_lines = []
    if context.grain:
        if context.grain.status == "estimated":
            grain_lines.append(f"Likely grain (inference, confidence {context.grain.confidence:.2f}): {context.grain.label}.")
            grain_lines.extend(f"  Evidence: {e.statement}." for e in context.grain.evidence)
        else:
            grain_lines.append("Likely grain: undetermined from available evidence.")
    warning_lines = ["WARNINGS"] if context.warnings else []
    for warning in context.warnings:
        warning_lines.append(f"- [{warning.severity}] {warning.message} {warning.implication}".strip())
    column_lines = ["COLUMNS"]
    ranked = sorted(context.columns, key=lambda c: (c.semantic.role not in {"identifier", "measure", "temporal"}, -len(c.warnings), c.name))
    for col in ranked:
        s = col.statistics
        column_lines.append(f"- {col.name} [{col.dtype}]: likely {col.semantic.role} ({col.semantic.confidence:.2f}); observed {s.distinct_count} distinct, {s.missing_rate:.1%} missing.")
        for e in col.semantic.evidence[:2]:
            column_lines.append(f"  Evidence: {e.statement}.")
        if s.minimum is not None or s.maximum is not None:
            column_lines.append(f"  Observed range: {s.minimum} to {s.maximum}.")
        for rec in col.recommendations:
            if rec.disposition in {"recommended", "caution"}:
                column_lines.append(f"  Suggested {rec.operation}: {rec.rationale}.")
    if context.relationships:
        column_lines.append("OBSERVED RELATIONSHIPS")
        for rel in context.relationships:
            column_lines.append(f"- {rel.source_column} -> {rel.target_column}: {rel.kind}, coverage {rel.coverage:.1%}.")
    lines = header + grain_lines + warning_lines + column_lines
    result = "\n".join(lines)
    if len(result) <= max_chars:
        return result
    marker = "[Context truncated.]"
    selected: list[str] = []
    for line in header + grain_lines + warning_lines + column_lines:
        candidate = "\n".join((*selected, line, marker))
        if len(candidate) <= max_chars:
            selected.append(line)
    if not selected:
        return marker[:max_chars]
    return "\n".join((*selected, marker))
