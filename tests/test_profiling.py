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
