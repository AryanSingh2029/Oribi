"""Candidate-key collection and separate grain selection."""

from itertools import combinations
import re

import pandas as pd

from ..models.evidence import Evidence, GrainContext, KeyCandidate

_MIN_SINGLE_UNIQUENESS = 0.98
_MAX_SINGLE_MISSING_RATE = 0.02
_MIN_COMPOSITE_UNIQUENESS = 0.95
_COMPOSITE_ROLES = {"identifier", "dimension", "categorical", "boolean"}


def detect_candidate_keys(
    frame: pd.DataFrame,
    names: list[str],
    stats: dict[str, object],
    roles: dict[str, str],
    max_columns: int = 8,
    max_exact_rows: int = 100_000,
) -> tuple[KeyCandidate, ...]:
    """Collect qualifying single and bounded pair candidates.

    Pair search considers at most ``max_columns`` non-unique identifier or
    dimension-like columns (28 pairs at the default limit of eight), and only
    runs below ``max_exact_rows``. This bounds exact combination scans while
    allowing all qualifying candidates in the searched set to be retained.
    """
    row_count = len(frame)
    if row_count == 0:
        return ()
    sample_support = min(1.0, (row_count / 100) ** 0.5)

    single_candidates: list[KeyCandidate] = []
    for name in sorted(names):
        column_stats = stats[name]
        rate = column_stats.distinct_count / row_count
        if (
            roles.get(name) == "identifier"
            and rate >= _MIN_SINGLE_UNIQUENESS
            and column_stats.missing_rate <= _MAX_SINGLE_MISSING_RATE
        ):
            evidence = Evidence(
                "single_column_uniqueness",
                f"{column_stats.distinct_count} distinct values across {row_count} rows",
                {"distinct_count": column_stats.distinct_count, "row_count": row_count},
            )
            single_candidates.append(KeyCandidate(
                (name,),
                "single",
                round(min(.95, .5 + (.6 + .4 * rate - .5) * sample_support), 2),
                rate,
                (evidence,),
            ))

    # A component already qualifying alone adds no useful composite candidate.
    single_columns = {candidate.columns[0] for candidate in single_candidates}
    potential = sorted(
        (
            name for name in names
            if name not in single_columns
            and _is_key_component(name, roles.get(name))
            and stats[name].distinct_count > 1
        ),
        key=lambda name: (-stats[name].distinct_rate, stats[name].missing_rate, name),
    )[:max_columns]

    pair_candidates: list[KeyCandidate] = []
    if row_count <= max_exact_rows:
        for pair in combinations(potential, 2):
            columns = tuple(sorted(pair))
            try:
                distinct_pairs = int(frame[list(columns)].drop_duplicates().shape[0])
            except (TypeError, ValueError):
                continue
            rate = distinct_pairs / row_count
            if rate >= _MIN_COMPOSITE_UNIQUENESS:
                base_confidence = min(.9, .55 + (rate - _MIN_COMPOSITE_UNIQUENESS) * 5)
                confidence = .5 + (base_confidence - .5) * sample_support
                evidence = Evidence(
                    "composite_uniqueness",
                    f"{distinct_pairs} distinct pairs across {row_count} rows",
                    {
                        "distinct_pairs": distinct_pairs,
                        "row_count": row_count,
                        "uniqueness_rate": rate,
                    },
                )
                pair_candidates.append(KeyCandidate(
                    columns, "composite", round(confidence, 2), rate, (evidence,),
                ))

    # Candidate retention does not depend on this ordering. Simpler keys are
    # listed first, with deterministic uniqueness/confidence/name tie-breakers.
    return tuple(sorted(
        (*single_candidates, *pair_candidates),
        key=lambda candidate: (
            len(candidate.columns),
            -candidate.uniqueness_rate,
            -candidate.confidence,
            candidate.columns,
        ),
    ))


def _is_key_component(name: str, role: str | None) -> bool:
    """Allow established key-like name hints when role votes are ambiguous."""
    if role in _COMPOSITE_ROLES:
        return True
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    return "id" in tokens or name.lower().endswith(("_key", "_code"))


def select_grain(candidates: tuple[KeyCandidate, ...]) -> GrainContext:
    """Choose one preferred grain from the retained candidates."""
    if not candidates:
        return GrainContext("undetermined", None, (), 0.0, ())

    # Preserve the existing preference for a single-column identifier when
    # available; candidate detection still retains every composite candidate.
    singles = [candidate for candidate in candidates if candidate.kind == "single"]
    pool = singles or [candidate for candidate in candidates if candidate.kind == "composite"]
    best = min(
        pool,
        key=lambda candidate: (
            -candidate.uniqueness_rate,
            -candidate.confidence,
            candidate.columns,
        ),
    )
    label = " × ".join(best.columns)
    return GrainContext("estimated", label, best.columns, best.confidence, best.evidence)


def detect_keys(
    frame: pd.DataFrame,
    names: list[str],
    stats: dict[str, object],
    roles: dict[str, str],
    max_columns: int = 8,
    max_exact_rows: int = 100_000,
) -> tuple[tuple[KeyCandidate, ...], GrainContext]:
    """Collect candidate keys, then independently select a grain hypothesis."""
    candidates = detect_candidate_keys(
        frame, names, stats, roles,
        max_columns=max_columns,
        max_exact_rows=max_exact_rows,
    )
    if len(frame) == 0:
        grain = GrainContext(
            "undetermined", None, (), 0.0,
            (Evidence("empty_dataset", "No rows are available to estimate a key or grain"),),
        )
    elif candidates:
        grain = select_grain(candidates)
    else:
        code = "key_search_bounded" if len(frame) > max_exact_rows else "no_strong_key"
        message = (
            "Composite key search was skipped because the DataFrame exceeds max_exact_rows"
            if len(frame) > max_exact_rows
            else "No single column or tested column pair was sufficiently unique"
        )
        grain = GrainContext(
            "undetermined", None, (), 0.0,
            (Evidence(code, message, {"row_count": len(frame), "max_exact_rows": max_exact_rows}),),
        )
    return candidates, grain
