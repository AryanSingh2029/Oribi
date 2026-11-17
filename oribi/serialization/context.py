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


def _relationship_lines(context: Any) -> list[str]:
    """Describe how columns relate to records, grain, and other columns."""
    lines = ["RELATIONSHIPS"]
    rendered_composite_keys: set[frozenset[str]] = set()
    for column in sorted(context.columns, key=lambda item: item.name):
        for relation in column.relationships:
            confidence = f"{relation.confidence:.0%} confidence"
            if relation.kind == "identifies_observations":
                evidence = relation.evidence[0].details if relation.evidence else {}
                card = column.cardinality
                unique = card.unique_count if card else evidence.get("distinct_count", column.statistics.distinct_count)
                rows = card.observed_count if card else evidence.get("row_count", context.profile.row_count)
                statement = (
                    "appears to uniquely identify observations"
                    if card and card.uniqueness_ratio == 1 and card.missing_count == 0
                    else "is a mostly-unique observation-key candidate"
                )
                strength = f"{relation.evidence_strength:.0%} uniqueness evidence; " if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {column.name} {statement} "
                    f"({unique:,}/{rows:,} values distinct; {strength}{confidence})"
                )
            elif relation.kind == "jointly_identifies_observations":
                composite_key = frozenset((column.name, *relation.related_columns))
                if composite_key in rendered_composite_keys:
                    continue
                rendered_composite_keys.add(composite_key)
                components = " + ".join((column.name, *relation.related_columns))
                evidence = relation.evidence[0].statement if relation.evidence else "observed joint uniqueness"
                strength = f"{relation.evidence_strength:.0%} uniqueness evidence; " if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {components} jointly identify an observation "
                    f"({evidence}; {strength}{confidence})"
                )
            elif relation.kind == "groups_observations":
                card = column.cardinality
                detail = relation.evidence[0].statement if relation.evidence else "repeated values"
                evidence_details = relation.evidence[0].details if relation.evidence else {}
                average_entities = evidence_details.get("average_entities_per_group")
                if relation.related_columns and average_entities is not None:
                    detail = f"about {average_entities:.1f} related observations per group"
                elif card and card.average_records_per_value is not None:
                    detail = (
                        f"{card.unique_count:,} unique values across {card.observed_count:,} "
                        f"non-missing observations; average {card.average_records_per_value:.2f} observations/value"
                    )
                if relation.related_columns:
                    detail += f"; maps observations to {', '.join(relation.related_columns)}"
                strength = f"; {relation.evidence_strength:.0%} evidence strength" if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {column.name} groups observations ({detail}{strength}; {confidence})"
                )
            elif relation.kind == "measure_at_grain":
                grain = (
                    " + ".join(relation.related_columns)
                    if relation.related_columns
                    else "the observed record grain"
                )
                evidence = relation.evidence[0].statement if relation.evidence else "numeric observations"
                lines.append(
                    f"- {column.name} is a quantitative measure at {grain} "
                    f"({evidence}; {confidence})"
                )
            elif relation.kind == "temporal_attribute":
                evidence = relation.evidence[0].statement if relation.evidence else "datetime-typed values"
                lines.append(
                    f"- {column.name} is a temporal attribute associated with observations "
                    f"({evidence}; {confidence})"
                )

    one_to_one_pairs = {
        frozenset((relation.source_column, relation.target_column))
        for relation in context.relationships if relation.kind == "one_to_one"
    }
    for relation in context.relationships:
        if relation.kind == "one_to_many":
            continue  # Already summarized from the grouping column.
        if relation.kind == "functional_dependency" and frozenset((relation.source_column, relation.target_column)) in one_to_one_pairs:
            continue
        if relation.kind == "one_to_one":
            evidence = relation.evidence[0].statement if relation.evidence else "observed value mapping"
            lines.append(
                f"- {relation.source_column} ↔ {relation.target_column}: observed one-to-one value mapping "
                f"({evidence}; {relation.coverage:.0%} coverage; {relation.confidence:.0%} confidence)"
            )
        else:
            evidence = relation.evidence[0].statement if relation.evidence else "observed value mapping"
            lines.append(
                f"- {relation.source_column} → {relation.target_column}: observed functional dependency "
                f"({evidence}; {relation.coverage:.0%} coverage; {relation.confidence:.0%} confidence)"
            )
    return lines if len(lines) > 1 else []


def _column_role_lines(context: Any) -> list[str]:
    labels = {
        "identifier": "identifier candidate",
        "dimension": "categorical dimension",
        "categorical": "categorical dimension",
        "measure": "numeric measure",
        "temporal": "temporal dimension",
        "boolean": "boolean/flag",
        "text": "text",
        "unknown": "unknown",
    }
    lines = ["COLUMN ROLES"]
    for column in sorted(context.columns, key=lambda item: item.name):
        role = labels.get(column.semantic.role, column.semantic.role)
        lines.append(f"- {column.name}: {role} ({column.semantic.confidence:.0%} confidence)")
    return lines if len(lines) > 1 else []


def _cardinality_lines(context: Any) -> list[str]:
    lines = ["CARDINALITY"]
    for column in sorted(context.columns, key=lambda item: item.name):
        card = column.cardinality
        if card is None:
            continue
        line = f"- {column.name}: {card.unique_count:,} unique / {card.row_count:,} rows"
        if card.uniqueness_ratio is None:
            line += "; no observed values"
        else:
            line += f" ({card.uniqueness_ratio:.0%} of observed rows unique)"
            line += f"; {card.observed_count:,} non-missing"
            line += f"; {card.missing_count:,} missing ({card.missing_rate:.0%})"
            line += f"; average {card.average_records_per_value:.2f} records/value"
            line += f" (min {card.minimum_records_per_value}, max {card.maximum_records_per_value})"
        lines.append(line)
    return lines if len(lines) > 1 else []


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
        grain_columns = " + ".join(context.grain.columns) or context.grain.label or "Unknown"
        if len(context.grain.columns) > 1:
            description = "candidate composite observation grain"
        elif any(column.name in context.grain.columns and column.logical_type == "temporal" for column in context.columns):
            description = "candidate temporal observation grain"
        else:
            description = "candidate observation grain"
        grain_lines.append(f"{grain_columns} \u00b7 {description} \u00b7 {context.grain.confidence:.0%} confidence")
    else:
        grain_lines.append("Unknown \u00b7 insufficient evidence")

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

    sections = [header, grain_lines]
    role_lines = _column_role_lines(context)
    if role_lines:
        sections.append(role_lines)
    if warning_lines:
        sections.append(warning_lines)
    relationship_lines = _relationship_lines(context)
    if relationship_lines:
        sections.append(relationship_lines)
    cardinality_lines = _cardinality_lines(context)
    if cardinality_lines:
        sections.append(cardinality_lines)
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
