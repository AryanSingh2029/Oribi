"""Human-readable context report."""

from typing import Any


def to_markdown(context: Any) -> str:
    profile = context.profile
    lines = [
        "# Dataset context",
        "",
        "## Dataset summary",
        "",
        f"- Rows: {profile.row_count:,}",
        f"- Columns: {profile.column_count:,}",
    ]
    if profile.duplicate_row_count is not None:
        lines.append(f"- Duplicate rows: {profile.duplicate_row_count:,}")

    lines += ["", "## Grain", ""]
    if context.grain and context.grain.status == "estimated":
        lines.append(f"**{context.grain.label}** · confidence {context.grain.confidence:.2f}")
    else:
        lines.append("Undetermined")
    if context.grain and context.grain.evidence:
        lines += ["", "<details>", "<summary>Grain inference evidence</summary>", ""]
        lines.extend(f"- {e.statement}" for e in context.grain.evidence)
        lines += ["", "</details>"]

    lines += ["", "## Columns", ""]
    for col in context.columns:
        s = col.statistics
        lines += [
            f"### {col.name}",
            "",
            f"**{col.dtype}** ({col.logical_type}) · **{col.semantic.role}** (confidence {col.semantic.confidence:.2f})",
            "",
            f"- Missing: {s.missing_rate:.1%} ({s.missing_count:,})",
            f"- Unique: {s.distinct_count:,}",
        ]
        if s.minimum is not None or s.maximum is not None:
            lines.append(f"- Range: {s.minimum} to {s.maximum}")
        aggregations = [
            rec.operation for rec in col.recommendations
            if rec.disposition in {"recommended", "caution"}
        ]
        lines.append(f"- Recommended aggregations: {', '.join(aggregations) if aggregations else 'None'}")
        if col.warnings:
            lines.extend(f"- Warning [{warning.severity}]: {warning.message}" for warning in col.warnings)
        if col.semantic.evidence:
            lines += ["", "<details>", "<summary>Inference evidence</summary>", ""]
            lines.extend(f"- {e.statement}" for e in col.semantic.evidence)
            lines += ["", "</details>"]
        lines.append("")

    if context.relationships:
        lines += ["## Relationships", ""]
        lines.extend(
            f"- {rel.source_column} → {rel.target_column}: {rel.kind}, coverage {rel.coverage:.1%}"
            for rel in context.relationships
        )
    if context.warnings:
        lines += ["", "## Analytical warnings", ""]
        for warning in context.warnings:
            lines.append(f"- **{warning.title}** ({warning.severity}): {warning.message}")
            if warning.implication:
                lines.append(f"  - Implication: {warning.implication}")
            if warning.recommendation:
                lines.append(f"  - Suggested: {warning.recommendation}")
    return "\n".join(lines).strip()
