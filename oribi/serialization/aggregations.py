"""Concise qualifications shared by the text exporters."""

from collections.abc import Iterable

from ..models.evidence import Recommendation


def aggregation_conditions(recommendations: Iterable[Recommendation], role: str) -> str:
    conditions = []
    for rec in recommendations:
        if rec.operation == "SUM" and role == "measure":
            conditions.append("SUM only for additive quantities")
        elif rec.operation == "MEAN" and role == "measure":
            if rec.disposition == "caution":
                conditions.append("check weighting and denominators before averaging")
            else:
                conditions.append("MEAN for roughly symmetric values")
        elif rec.operation == "GROUP BY" and rec.disposition == "caution":
            conditions.append("group only if many small groups are useful")
    return " (" + " / ".join(conditions) + ")" if conditions else ""
