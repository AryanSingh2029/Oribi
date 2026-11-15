"""Human-readable context report."""

from typing import Any


def to_markdown(context: Any) -> str:
    profile = context.profile
    lines = ["# Dataset context", "", f"- Rows: {profile.row_count:,}", f"- Columns: {profile.column_count:,}"]
    if profile.duplicate_row_count is not None:
        lines.append(f"- Duplicate rows: {profile.duplicate_row_count:,}")
    lines += ["", "## Likely grain", ""]
    if context.grain and context.grain.status == "estimated":
        lines.append(f"{context.grain.label} (confidence {context.grain.confidence:.2f})")
        lines.extend(f"- {e.statement}" for e in context.grain.evidence)
    else:
        lines.append("Undetermined")
    lines += ["", "## Columns", ""]
    for col in context.columns:
        s = col.statistics
        lines += [f"### {col.name}", f"- Type: {col.dtype} ({col.logical_type})", f"- Likely role: {col.semantic.role} ({col.semantic.confidence:.2f})", f"- Missing: {s.missing_count:,} ({s.missing_rate:.1%})", f"- Distinct: {s.distinct_count:,} ({s.distinct_rate:.1%})"]
        if s.minimum is not None or s.maximum is not None:
            lines.append(f"- Range: {s.minimum} to {s.maximum}")
        for rec in col.recommendations:
            lines.append(f"- {rec.disposition.title()}: {rec.operation} — {rec.rationale}")
        for warning in col.warnings:
            lines.append(f"- Warning [{warning.severity}]: {warning.message}")
        lines.append("")
    if context.warnings:
        lines += ["## Analytical warnings", ""]
        for warning in context.warnings:
            lines.append(f"- **{warning.title}** ({warning.severity}): {warning.message}")
            if warning.implication:
                lines.append(f"  - Implication: {warning.implication}")
            if warning.recommendation:
                lines.append(f"  - Suggested: {warning.recommendation}")
    return "\n".join(lines).strip()
