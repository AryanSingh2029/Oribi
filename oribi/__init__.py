"""Build structured analytical context from pandas DataFrames."""

from .api import ProfileOptions, inspect
from .exceptions import InvalidInputError, OribiError
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
    "ColumnContext", "DatasetContext", "Evidence", "GrainContext", "InvalidInputError",
    "KeyCandidate", "ProfileOptions", "Recommendation", "RelationshipContext",
    "OribiError", "SemanticContext", "StatisticsContext", "Warning", "inspect",
]

__version__ = "0.1.0"
