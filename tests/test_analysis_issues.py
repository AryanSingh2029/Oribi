import json
from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from oribi import ProfileOptions, Warning, inspect


@pytest.mark.parametrize(
    "values,dtype,expected",
    [
        (["same", "same", None], "string", "constant_column"),
        ([1] * 19 + [2, None], "Int64", "near_constant_column"),
        ([True] * 18 + [False], "boolean", None),
        ([True, 1, True], "object", None),
        ([1, None], "Int64", None),
        ([None, None], "object", None),
        ([], "float64", None),
    ],
)
def test_constancy_counts_only_observed_values(values, dtype, expected):
    (column,) = inspect(pd.Series(values, dtype=dtype, name="value").to_frame()).columns
    evidence = column.cardinality.constancy
    if expected is None:
        assert evidence is None
    else:
        observed = len(values) - values.count(None)
        assert evidence.code == expected
        assert evidence.details == {
            "observed_count": observed,
            "dominant_count": observed if expected == "constant_column" else observed - 1,
            "dominant_share": 1.0 if expected == "constant_column" else .95,
            "missing_count": values.count(None),
        }


def test_constancy_does_not_depend_on_sampled_values():
    frame = pd.DataFrame({"quantity": [10] * 190 + list(range(11, 21))})
    (column,) = inspect(frame, options=ProfileOptions(top_k=1)).columns
    assert not column.statistics.top_values
    assert column.cardinality.constancy.details["dominant_share"] == .95


def test_sparse_duplicates_match_dense_without_mutating_the_input():
    values = pd.Series(pd.arrays.SparseArray([0.0, 0.0, None, None, 1.0]), name="amount")
    frame = values.to_frame()
    sparse = inspect(frame)
    dense = inspect(values.sparse.to_dense().to_frame())
    assert sparse.profile.duplicate_row_count == dense.profile.duplicate_row_count == 2
    assert sparse.columns[0].dtype == str(values.dtype)
    pd.testing.assert_frame_equal(frame, values.to_frame())


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_iso_temporal_roles_are_evidence_backed_and_preserve_storage(dtype):
    values = pd.Series(["2026-01-01T01:02:03.1Z", "2026-01-01T01:02:03.123456789+01:00", None], dtype=dtype)
    frame = values.to_frame("observed")
    context = inspect(frame, options=ProfileOptions(max_exact_rows=1))
    (column,) = context.columns
    assert column.dtype == str(values.dtype)
    assert column.logical_type == "text"
    assert column.semantic.role == "temporal"
    evidence = next(e for e in column.semantic.evidence if e.code == "iso_temporal_values")
    assert evidence.details == {"observed_count": 2, "distinct_count": 2}
    assert not context.candidate_keys
    assert context.grain.status == "undetermined"
    assert column.statistics.date_range is None
    assert not any(w.code == "potential_temporal_gap" for w in context.warnings)
    pd.testing.assert_frame_equal(frame, values.to_frame("observed"))


def test_python_date_objects_and_identifier_conflicts():
    values = pd.Series([date(2026, 1, 1), datetime(2026, 1, 2, tzinfo=timezone.utc)], dtype=object)
    context = inspect(pd.DataFrame({"day": values, "batch_id": ["2026-01-01", "2026-01-02"]}))
    day, batch = context.columns
    assert day.semantic.role == "temporal"
    assert any(e.code == "temporal_object_values" for e in day.semantic.evidence)
    assert batch.semantic.role == "unknown"
    assert {e.code for e in batch.semantic.evidence} >= {"name_identifier_hint", "iso_temporal_values"}
    assert not context.candidate_keys


@pytest.mark.parametrize("values", [
    ["20260101", "20260102"],
    ["01/02/2026", "01/03/2026"],
    ["2026-01-01", "2026-02-30"],
    ["2026-01-01", "2026-03-02T12:00:00+00:75"],
    ["2026-01-01", "2026-01-01", None],
    ["2026-01-01"] * 101 + ["2026-01-02", "unavailable"],
    [date(2026, 1, 1), "2026-01-02"],
])
def test_ambiguous_or_invalid_values_do_not_infer_temporal_roles(values):
    for ordered in (values, list(reversed(values))):
        (column,) = inspect(pd.Series(ordered, dtype=object, name="value").to_frame()).columns
        assert column.semantic.role != "temporal"
        assert not any(e.code in {"iso_temporal_values", "temporal_object_values"} for e in column.semantic.evidence)


def test_mixed_objects_are_classified_the_same_above_limit_in_any_order():
    values = pd.Series(["x"] * 105 + [1, None], dtype=object)
    options = ProfileOptions(max_exact_rows=100)
    orders = [values, values.iloc[::-1], values.sample(frac=1, random_state=7)]
    columns = [inspect(order.to_frame("value"), options=options).columns[0] for order in orders]
    assert all(column.logical_type == "unknown" for column in columns)
    assert all(column.semantic == columns[0].semantic for column in columns)
    assert all(column.statistics.missing_count == 1 for column in columns)
    assert columns[0].cardinality.constancy is None


def test_exact_row_limit_is_reported_even_when_grain_is_inferred():
    frame = pd.DataFrame({"row_id": range(4), "amount": [1.5, 2.0, 3.0, 4.0]})
    at_limit = inspect(frame, options=ProfileOptions(max_exact_rows=4))
    disabled = inspect(frame, options=ProfileOptions(max_exact_rows=4, include_quantiles=False))
    above = inspect(frame, options=ProfileOptions(max_exact_rows=3))

    assert at_limit.columns[1].statistics.quantiles
    assert not disabled.columns[1].statistics.quantiles
    assert not above.columns[1].statistics.quantiles
    assert not at_limit.profile.exact_row_limit_exceeded
    assert not disabled.profile.exact_row_limit_exceeded
    assert above.profile.exact_row_limit_exceeded
    assert above.grain.columns == at_limit.grain.columns == ("row_id",)
    assert above.to_dict()["profile"]["exact_row_limit_exceeded"] is True
    assert "Exact row limit exceeded." in above.to_context(max_chars=100)
    assert len(above.to_context(max_chars=100)) <= 100


def test_summary_prioritizes_three_distinct_messages_and_preserves_grain():
    context = inspect(pd.DataFrame({"order_id": range(4)}))
    assert "Key findings" not in context.summary()
    assert "Likely grain: order_id" in context.summary()
    low = Warning("low", "low", "Low title", "Low priority")
    medium = Warning("medium", "medium", "Medium title", "Medium " + "details " * 100)
    high = Warning("high", "high", "High title", "High priority")
    critical = Warning("critical", "critical", "Critical title", "Critical priority")
    summary = replace(context, warnings=(low, high, medium, high, critical)).summary()
    assert "Likely grain: order_id" in summary
    assert summary.index("[critical]") < summary.index("[high]") < summary.index("[medium]")
    assert summary.count("High priority") == 1
    assert "Low priority" not in summary
    assert len(summary) < 450


def test_observed_relationship_cardinality_and_coverage_across_exports():
    frame = pd.DataFrame({
        "line_id": range(6),
        "store": pd.Categorical(["a", "a", "b", "b", "c", "c"], categories=["a", "b", "c", "unused"]),
        "region": ["east"] * 4 + ["west"] * 2,
        "store_label": ["A", "A", "B", "B", "C", "C"],
    })
    context = inspect(frame)
    indexed = {(r.source_column, r.target_column, r.kind): r for r in context.relationships}
    dependency = indexed["store", "region", "functional_dependency"]
    one_to_many = indexed["store", "line_id", "one_to_many"]
    assert dependency.mapping_cardinality == "many_to_one"
    assert dependency.coverage == 1 and dependency.paired_row_count == 6
    assert dependency.evidence[0].details["source_distinct"] == 3
    assert dependency.evidence[0].details["target_distinct"] == 2
    assert ("region", "store", "functional_dependency") not in indexed
    assert one_to_many.mapping_cardinality == "one_to_many"
    assert one_to_many.coverage == 1 and one_to_many.paired_row_count == 6
    assert one_to_many.evidence[0].details["minimum_entities_per_group"] == 2
    assert one_to_many.evidence[0].details["average_entities_per_group"] == 2
    bijections = [r for r in context.relationships if r.kind == "one_to_one"]
    assert any(
        {r.source_column, r.target_column} == {"store", "store_label"}
        and r.mapping_cardinality == "one_to_one" and r.paired_row_count == 6
        for r in bijections
    )
    assert "many-to-one" in context.to_markdown()
    assert "many-to-one" in context.to_context()
    assert "one-to-many" in context.to_context()
    assert {r["mapping_cardinality"] for r in context.to_dict()["relationships"]} >= {
        "many_to_one", "one_to_many", "one_to_one",
    }


@pytest.mark.parametrize("unmatched", ["conflict", "missing"])
def test_incomplete_or_conflicting_dependencies_are_approximate(unmatched):
    mapping = [(f"store-{i}", f"region-{i}") for i in range(50)]
    pairs = mapping * 2
    pairs[-1] = (pairs[-1][0], "region-0" if unmatched == "conflict" else None)
    context = inspect(pd.DataFrame(pairs, columns=["store", "region"]))
    relation = next(r for r in context.relationships if r.source_column == "store" and r.target_column == "region")
    assert relation.kind == "approximate_mapping"
    assert relation.mapping_cardinality is None
    assert relation.coverage == (.98 if unmatched == "conflict" else 1.0)
    assert relation.violation_count == (1 if unmatched == "conflict" else 0)
    assert relation.missing_pair_count == (0 if unmatched == "conflict" else 1)


def test_integrated_findings_across_renderers_and_row_orders():
    secret = "private-constant-value"
    frame = pd.DataFrame({
        "row_id": range(20),
        "batch": pd.Categorical([secret] * 19 + ["other"], categories=[secret, "other", "unused"]),
        "logged": ["2026-01-01", "2026-01-02"] * 10,
        "amount": [20.5, None, None, None, 10.0] * 4,
        "mixed": pd.Series(["code"] * 19 + [7], dtype=object),
    })
    original = frame.copy(deep=True)
    options = ProfileOptions(max_exact_rows=10)
    first = inspect(frame, options=options)
    reversed_rows = inspect(frame.iloc[::-1].reset_index(drop=True), options=options)
    for context in (first, reversed_rows):
        summary = context.summary()
        assert "Likely grain: row_id" in summary and "amount is missing" in summary
        assert "Exact row limit exceeded." in summary
        for rendered in (context.to_markdown(), context.to_context()):
            assert "Near-constant" in rendered and "ISO dates or timestamps" in rendered
            assert "Exact row limit exceeded." in rendered
            assert secret not in rendered
        data = context.to_dict()
        assert json.loads(context.to_json()) == data
        by_name = {column["name"]: column for column in data["columns"]}
        assert by_name["batch"]["cardinality"]["constancy"]["details"]["dominant_count"] == 19
        assert by_name["logged"]["semantic"]["role"] == "temporal"
        assert by_name["mixed"]["logical_type"] == "unknown"
        assert data["profile"]["exact_row_limit_exceeded"] is True
        assert data["relationships"] == []
        assert secret not in context.to_json()
    pd.testing.assert_frame_equal(frame, original)
