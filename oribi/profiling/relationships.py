"""Conservative within-frame column dependency detection."""

import pandas as pd

from ..models.evidence import Evidence, RelationshipContext


def detect_relationships(frame: pd.DataFrame, roles: dict[str, str], max_rows: int = 100_000) -> tuple[RelationshipContext, ...]:
    """Find near-complete one-to-one value mappings among low-cardinality columns."""
    if len(frame) == 0 or len(frame) > max_rows:
        return ()
    eligible = [name for name, role in roles.items() if role in {"identifier", "dimension", "categorical"}]
    eligible = [name for name in eligible if frame[name].nunique(dropna=True) <= min(500, max(20, len(frame) // 2))]
    results: list[RelationshipContext] = []
    for source in eligible:
        for target in eligible:
            if source == target:
                continue
            pairs = frame[[source, target]].dropna().drop_duplicates()
            source_values = frame[source].dropna().nunique()
            if source_values < 2:
                continue
            unique_mapping = int(pairs.groupby(source, dropna=True)[target].nunique().le(1).sum())
            coverage = unique_mapping / source_values
            target_values = frame[target].dropna().nunique()
            if coverage >= .98 and target_values > 1 and source_values <= target_values:
                results.append(RelationshipContext(
                    source, target, "functional_dependency", coverage,
                    round(min(.95, .65 + coverage * .3), 2),
                    (Evidence("value_mapping", f"{coverage:.1%} of observed {source} values map to one {target} value", {"source_distinct": int(source_values), "target_distinct": int(target_values), "coverage": coverage}),),
                ))
    # Keep only one orientation per pair to avoid duplicate, reversed claims.
    deduped: dict[frozenset[str], RelationshipContext] = {}
    for relation in results:
        key = frozenset((relation.source_column, relation.target_column))
        current = deduped.get(key)
        if current is None or relation.source_column < current.source_column:
            deduped[key] = relation
    return tuple(sorted(deduped.values(), key=lambda r: (r.source_column, r.target_column)))
