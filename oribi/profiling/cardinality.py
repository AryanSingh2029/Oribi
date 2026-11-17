"""Observed per-value cardinality summaries."""

import pandas as pd

from ..models.evidence import CardinalityContext


def profile_cardinality(series: pd.Series, row_count: int) -> CardinalityContext:
    """Describe observed frequencies without assigning business meaning.

    Missing values are excluded from the frequency distribution; row_count
    still includes them so callers can see the coverage of the column.
    """
    observed = series.dropna()
    try:
        counts = observed.value_counts(dropna=True, sort=False)
    except (TypeError, ValueError):
        # Match AnalysisSession's fallback for unhashable object values.
        counts = observed.astype(str).value_counts(dropna=True, sort=False)
    counts = counts[counts > 0]
    observed_count = int(counts.sum())
    unique_count = int(len(counts))
    if not unique_count:
        ratio = average = None
        minimum = maximum = None
    else:
        ratio = unique_count / observed_count
        average = observed_count / unique_count
        minimum = int(counts.min())
        maximum = int(counts.max())
    return CardinalityContext(
        row_count=row_count,
        observed_count=observed_count,
        unique_count=unique_count,
        uniqueness_ratio=ratio,
        average_records_per_value=average,
        minimum_records_per_value=minimum,
        maximum_records_per_value=maximum,
        missing_count=row_count - observed_count,
        missing_rate=(row_count - observed_count) / row_count if row_count else 0.0,
    )
