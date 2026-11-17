"""Public inspection API and pipeline orchestration."""

from dataclasses import dataclass

import pandas as pd

from .exceptions import InvalidInputError
from .models.context import ColumnContext, DatasetContext
from .models.evidence import Warning
from .models.profile import DatasetProfile
from .profiling.columns import profile_column
from .profiling.cardinality import profile_cardinality
from .profiling.observations import AnalysisSession
from .profiling.relationships import detect_relationships, infer_column_relationships
from .profiling.temporal import temporal_gap_warnings
from .reasoning.keys import detect_keys
from .reasoning.rules import dataset_warnings, reason_column
from .reasoning.semantic import infer_semantic


@dataclass(frozen=True)
class ProfileOptions:
    """Controls profiling work while keeping results deterministic."""

    top_k: int = 10
    include_quantiles: bool = True
    max_exact_rows: int = 100_000
    max_key_columns: int = 8

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_exact_rows < 1 or self.max_key_columns < 2:
            raise ValueError("max_exact_rows must be positive and max_key_columns at least 2")


def inspect(df: pd.DataFrame, *, options: ProfileOptions | None = None) -> DatasetContext:
    """Profile a pandas DataFrame and return its analytical context snapshot.

    The returned context contains no reference to ``df``. Object columns that
    appear datetime-like are not parsed automatically.
    """
    options = options or ProfileOptions()
    if not isinstance(options, ProfileOptions):
        raise InvalidInputError("options must be a ProfileOptions instance")
    session = AnalysisSession(df, options)
    nrows, ncols = df.shape
    try:
        duplicates = int(df.duplicated().sum())
    except (TypeError, ValueError):
        duplicates = None
    duplicate_rate = (duplicates / nrows) if duplicates is not None and nrows else (0.0 if duplicates == 0 else None)
    names = list(df.columns)
    observations = {name: session.observe(name) for name in names}
    stats = {name: profile_column(obs, nrows, options.top_k, options.include_quantiles and nrows <= options.max_exact_rows) for name, obs in observations.items()}
    semantics = {
        name: infer_semantic(
            name,
            observations[name].logical_type,
            stats[name],
            nrows,
            is_float=pd.api.types.is_float_dtype(observations[name].series.dtype),
        )
        for name in names
    }
    key_candidates, grain = detect_keys(
        df, names, stats, {name: semantics[name].role for name in names},
        max_columns=options.max_key_columns, max_exact_rows=options.max_exact_rows,
    )
    cardinalities = {
        name: profile_cardinality(observations[name].series, nrows)
        for name in names
    }
    role_map = {name: semantics[name].role for name in names}
    relationships = detect_relationships(
        df,
        role_map,
        max_rows=options.max_exact_rows,
        candidate_keys=key_candidates,
        grain=grain,
    )
    column_relationships = infer_column_relationships(
        cardinalities,
        semantics,
        {name: observations[name].logical_type for name in names},
        key_candidates,
        grain,
        relationships,
    )
    columns = []
    for name in names:
        obs = observations[name]
        recommendations, warnings = reason_column(name, obs.logical_type, semantics[name], stats[name])
        if obs.logical_type == "temporal":
            warnings += temporal_gap_warnings(obs.series, name)
        columns.append(ColumnContext(
            name,
            str(obs.series.dtype),
            obs.logical_type,
            stats[name],
            semantics[name],
            recommendations,
            warnings,
            cardinalities[name],
            column_relationships.get(name, ()),
        ))
    global_warnings = list(dataset_warnings(duplicates, nrows))
    for col in columns:
        global_warnings.extend(col.warnings)
    if duplicates is None:
        global_warnings.append(Warning(
            "duplicate_check_unavailable", "low", "Duplicate-row check unavailable",
            "Pandas could not compare rows with the available cell values.",
        ))
    return DatasetContext(
        profile=DatasetProfile(nrows, ncols, duplicates, duplicate_rate),
        columns=tuple(columns),
        candidate_keys=key_candidates,
        grain=grain,
        relationships=relationships,
        warnings=tuple(global_warnings),
    )
