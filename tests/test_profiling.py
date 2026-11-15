import pandas as pd

from datacontext import inspect


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
    dates = pd.to_datetime(values)
    context = inspect(pd.DataFrame({"event_time": dates}))
    column = next(c for c in context.columns if c.name == "event_time")
    return context, column, [w for w in column.warnings if w.code == "potential_temporal_gap"]


def test_daily_series_reports_single_potential_gap():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
    assert len(warnings) == 1
    details = warnings[0].evidence[0].details
    assert details["cadence"] == "1 day"
    assert details["expected_timestamps"] == ["2026-01-03T00:00:00"]
    assert details["observed_before"] == "2026-01-02T00:00:00"
    assert details["observed_after"] == "2026-01-04T00:00:00"


def test_daily_series_reports_multiple_potential_gaps():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-06", "2026-01-07"])
    assert len(warnings) == 1
    details = warnings[0].evidence[0].details
    assert details["missing_intervals"] == 3
    assert details["expected_timestamps"] == [
        "2026-01-03T00:00:00", "2026-01-04T00:00:00", "2026-01-05T00:00:00",
    ]


def test_weekly_series_reports_missing_week():
    _, _, warnings = _temporal_warnings(["2026-01-05", "2026-01-12", "2026-01-26", "2026-02-02"])
    assert len(warnings) == 1
    details = warnings[0].evidence[0].details
    assert details["cadence"] == "7 days"
    assert details["expected_timestamps"] == ["2026-01-19T00:00:00"]


def test_irregular_timestamps_do_not_report_gap():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-08", "2026-01-09", "2026-01-17", "2026-01-25"])
    assert warnings == []


def test_duplicates_do_not_distort_cadence_or_gap():
    _, column, warnings = _temporal_warnings([
        "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-05",
    ])
    assert len(warnings) == 1
    assert warnings[0].evidence[0].details["expected_timestamps"] == ["2026-01-04T00:00:00"]
    assert column.statistics.minimum == pd.Timestamp("2026-01-01")
    assert column.statistics.maximum == pd.Timestamp("2026-01-05")


def test_short_and_ambiguous_series_do_not_report_gap():
    _, _, short_warnings = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-05"])
    _, _, ambiguous_warnings = _temporal_warnings([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-07",
    ])
    assert short_warnings == []
    assert ambiguous_warnings == []


def test_temporal_gap_warning_is_serialized():
    context, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
    assert warnings[0] in context.warnings
    serialized = context.to_dict()
    assert any(warning["code"] == "potential_temporal_gap" for warning in serialized["warnings"])
