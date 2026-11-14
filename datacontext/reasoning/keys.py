"""Candidate key and grain estimation."""

from itertools import combinations

import pandas as pd

from ..models.evidence import Evidence, GrainContext, KeyCandidate


def detect_keys(frame: pd.DataFrame, names: list[str], stats: dict[str, object], roles: dict[str, str], max_columns: int = 8, max_exact_rows: int = 100_000) -> tuple[tuple[KeyCandidate, ...], GrainContext]:
    n = len(frame)
    if n == 0:
        return (), GrainContext("undetermined", None, (), 0.0, (Evidence("empty_dataset", "No rows are available to estimate a key or grain"),))
    candidates: list[KeyCandidate] = []
    for name in names:
        s = stats[name]
        rate = s.distinct_count / n
        if roles.get(name) == "identifier" and rate >= .98 and s.missing_rate <= .02:
            candidates.append(KeyCandidate((name,), "single", round(min(.99, .6 + .4 * rate), 2), rate, (Evidence("single_column_uniqueness", f"{s.distinct_count} distinct values across {n} rows", {"distinct_count": s.distinct_count, "row_count": n}),)))
    if candidates:
        best = candidates[0]
        grain = GrainContext("estimated", best.columns[0], best.columns, best.confidence, best.evidence)
        return tuple(candidates[:10]), grain
    potential = sorted(
        (name for name in names if roles.get(name) in {"identifier", "dimension", "categorical", "boolean"}),
        key=lambda name: (stats[name].distinct_rate, -stats[name].missing_rate),
        reverse=True,
    )[:max_columns]
    pair_candidates = []
    pair_iter = combinations(potential, 2) if n <= max_exact_rows else ()
    for a, b in pair_iter:
        try:
            distinct = int(frame[[a, b]].drop_duplicates().shape[0])
        except (TypeError, ValueError):
            continue
        rate = distinct / n
        if rate >= .95:
            confidence = min(.9, .55 + (rate - .95) * 5)
            ev = (Evidence("composite_uniqueness", f"{distinct} distinct pairs across {n} rows", {"distinct_pairs": distinct, "row_count": n, "uniqueness_rate": rate}),)
            pair_candidates.append(KeyCandidate((a, b), "composite", round(confidence, 2), rate, ev))
    pair_candidates.sort(key=lambda k: (-k.uniqueness_rate, -k.confidence, k.columns))
    candidates.extend(pair_candidates[:5])
    if pair_candidates:
        best = pair_candidates[0]
        label = " × ".join(best.columns)
        grain = GrainContext("estimated", label, best.columns, best.confidence, best.evidence)
    else:
        code = "key_search_bounded" if n > max_exact_rows else "no_strong_key"
        message = "Composite key search was skipped because the DataFrame exceeds max_exact_rows" if n > max_exact_rows else "No single column or tested column pair was sufficiently unique"
        grain = GrainContext("undetermined", None, (), 0.0, (Evidence(code, message, {"row_count": n, "max_exact_rows": max_exact_rows}),))
    return tuple(candidates), grain
