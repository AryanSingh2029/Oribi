"""Build structured analytical context from pandas DataFrames."""

from .api import ProfileOptions, inspect
from .models import (
    ColumnContext,
    DatasetContext,
    Evidence,
    GrainContext,
    KeyCandidate,
    Recommendation,
    RelationshipContext,
    SemanticContext,
    StatisticsContext,
    Warning,
)

__all__ = [
    "ColumnContext", "DatasetContext", "Evidence", "GrainContext",
    "KeyCandidate", "ProfileOptions", "Recommendation", "RelationshipContext",
    "SemanticContext", "StatisticsContext", "Warning", "inspect",
]

__version__ = "0.1.0"
