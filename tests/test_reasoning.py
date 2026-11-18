import pandas as pd

from oribi import inspect


def test_recommendations_and_warnings(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    assert {r.operation for r in revenue.recommendations} >= {"SUM", "MEAN", "MEDIAN"}
    assert any(w.code == "negative_measure_values" for w in revenue.warnings)
    assert all(w.code and w.evidence is not None for w in context.warnings)


def test_detects_deterministic_value_mapping():
    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"], "amount": [1, 2, 3, 4]})
    context = inspect(frame)
    assert any(r.kind == "functional_dependency" for r in context.relationships)


def test_missing_unique_identifiers_warn_for_missingness_only():
    column = inspect(pd.DataFrame({"account_id": [1, 2, 3, 4, None]})).columns[0]
    assert column.semantic.role == "identifier"
    assert any(w.code == "high_missingness" for w in column.warnings)
    assert not any(w.code == "non_unique_identifier" for w in column.warnings)


def test_null_only_identifiers_do_not_warn_about_duplicates():
    column = inspect(pd.DataFrame({"account_id": [None, None, None]})).columns[0]
    assert any(w.code == "high_missingness" for w in column.warnings)
    assert not any(w.code == "non_unique_identifier" for w in column.warnings)


def test_observed_duplicate_identifiers_still_warn():
    column = inspect(pd.DataFrame({"account_id": [1, 2, 2, 3, None]})).columns[0]
    assert column.semantic.role == "identifier"
    assert any(w.code == "non_unique_identifier" for w in column.warnings)
