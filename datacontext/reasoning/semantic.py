"""Composable rules for likely column roles."""

from ..models.evidence import Evidence, SemanticContext
import re


def infer_semantic(name: str, logical_type: str, stats: object, row_count: int) -> SemanticContext:
    votes: dict[str, float] = {}
    evidence: list[Evidence] = []
    lname = name.lower()
    distinct = stats.distinct_count
    rate = stats.distinct_rate
    tokens = set(re.split(r"[^a-z0-9]+", lname))
    if "id" in tokens or lname.endswith(("_key", "_code")):
        votes["identifier"] = votes.get("identifier", 0) + .72
        evidence.append(Evidence("name_identifier_hint", "Column name suggests an identifier"))
    if row_count and rate >= .98 and distinct > 1:
        votes["identifier"] = votes.get("identifier", 0) + (.75 if rate >= .98 else .62)
        evidence.append(Evidence("mostly_unique", f"{distinct} distinct non-null values across {row_count} rows", {"distinct_count": distinct, "row_count": row_count, "distinct_rate": rate}))
    if logical_type == "numeric":
        votes["measure"] = votes.get("measure", 0) + .48
        evidence.append(Evidence("numeric_type", "Column has a numeric dtype"))
        if any(w in lname for w in ("amount", "price", "revenue", "sales", "quantity", "count", "total", "cost", "rate")):
            votes["measure"] += .2
            evidence.append(Evidence("name_measure_hint", "Column name suggests a quantitative measure"))
    elif logical_type == "temporal":
        votes["temporal"] = .86
        evidence.append(Evidence("datetime_dtype", "Column has a datetime dtype"))
    elif logical_type == "boolean":
        votes["boolean"] = .9
        evidence.append(Evidence("boolean_dtype", "Column has a boolean dtype"))
    elif logical_type == "categorical":
        votes["categorical"] = .78
        evidence.append(Evidence("categorical_dtype", "Column has a categorical dtype"))
    elif logical_type == "text":
        if rate > .5 and distinct > 20:
            votes["text"] = .62
            evidence.append(Evidence("high_cardinality_text", "Text values have relatively high cardinality"))
        else:
            votes["dimension"] = .62
            evidence.append(Evidence("text_dimension", "Text values are reusable across rows"))
    if logical_type == "numeric" and distinct <= min(20, max(2, row_count // 10)):
        votes["dimension"] = max(votes.get("dimension", 0), .55)
        evidence.append(Evidence("low_cardinality_numeric", "Numeric values have low cardinality"))
    if not votes:
        return SemanticContext("unknown", 0.0, tuple(evidence))
    role, score = max(votes.items(), key=lambda item: (item[1], item[0]))
    if len(votes) > 1:
        ordered = sorted(votes.values(), reverse=True)
        if ordered[0] - ordered[1] < .12:
            return SemanticContext("unknown", round(min(ordered[0], .5), 2), tuple(evidence))
    return SemanticContext(role, round(min(score, .98), 2), tuple(evidence))
