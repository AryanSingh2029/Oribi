import pandas as pd

from oribi import inspect


def test_numeric_and_temporal_profile(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    dates = next(c for c in context.columns if c.name == "order_date")
    assert revenue.statistics.minimum == -2
    assert revenue.statistics.negative_count == 1
    assert revenue.statistics.zero_count == 1
    assert dates.logical_type == "temporal"
    assert dates.statistics.minimum.year == 2024


def test_missing_duplicates_and_empty():
    frame = pd.DataFrame({"x": [1, 1, None], "label": ["a", "a", "b"]})
    context = inspect(frame)
    assert context.profile.duplicate_row_count == 1
    assert context.columns[0].statistics.missing_count == 1
    empty = inspect(pd.DataFrame({"x": pd.Series(dtype="float64")}))
    assert empty.profile.row_count == 0
    assert empty.columns[0].statistics.distinct_count == 0


def test_all_null_and_mixed_columns():
    context = inspect(pd.DataFrame({"empty": [None, None], "mixed": ["a", 1]}))
    assert context.columns[0].logical_type == "unknown"
    assert context.columns[0].statistics.missing_rate == 1
    assert context.columns[1].logical_type == "unknown"


def _temporal_warnings(values):
    context = inspect(pd.DataFrame({"event_time": pd.to_datetime(values)}))
    column = next(c for c in context.columns if c.name == "event_time")
    return context, column, [w for w in column.warnings if w.code == "potential_temporal_gap"]


def test_regular_daily_series_with_gap_has_cadence_and_interval_evidence():
    context, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
    assert len(warnings) == 1
    evidence = warnings[0].evidence[0]
    assert evidence.details["cadence"] == "1 day"
    assert evidence.details["expected_timestamps"] == ("2026-01-03T00:00:00",)
    assert warnings[0] in context.warnings


def test_multiple_missing_intervals_are_reported():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-06", "2026-01-07"])
    assert warnings[0].evidence[0].details["missing_intervals"] == 3


def test_weekly_cadence_is_supported():
    _, _, warnings = _temporal_warnings(["2026-01-05", "2026-01-12", "2026-01-26", "2026-02-02"])
    assert warnings[0].evidence[0].details["cadence"] == "7 days"


def test_irregular_series_does_not_generate_gap_warning():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-08", "2026-01-09", "2026-01-17", "2026-01-25"])
    assert warnings == []


def test_duplicates_do_not_distort_cadence_and_gap():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-05"])
    assert len(warnings) == 1
    assert warnings[0].evidence[0].details["expected_timestamps"] == ("2026-01-04T00:00:00",)


def test_short_and_ambiguous_series_do_not_generate_gap_warning():
    _, _, short = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-05"])
    _, _, ambiguous = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-07"])
    assert short == []
    assert ambiguous == []


def test_timezone_aware_and_nullable_dtypes_profile():
    frame = pd.DataFrame({
        "at": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
        "count": pd.Series([1, None, 3, 4], dtype="Int64"),
        "active": pd.Series([True, None, False, True], dtype="boolean"),
        "category": pd.Series(["a", "b", "a", "c"], dtype="category"),
    })
    context = inspect(frame)
    columns = {column.name: column for column in context.columns}
    assert columns["at"].logical_type == "temporal"
    assert columns["at"].statistics.minimum.tz is not None
    assert columns["count"].statistics.missing_count == 1
    assert columns["active"].logical_type == "boolean"
    assert columns["category"].logical_type == "categorical"


def test_high_cardinality_categorical_column_has_warning():
    frame = pd.DataFrame({"group": pd.Series([f"g{i}" for i in range(110)], dtype="category")})
    context = inspect(frame)
    group = context.columns[0]
    assert group.logical_type == "categorical"
    assert any(w.code == "high_cardinality_category" for w in group.warnings)
