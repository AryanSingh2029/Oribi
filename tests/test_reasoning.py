from datacontext import inspect


def test_recommendations_and_warnings(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    assert {r.operation for r in revenue.recommendations} >= {"SUM", "MEAN", "MEDIAN"}
    assert any(w.code == "negative_measure_values" for w in revenue.warnings)
    assert all(w.code and w.evidence is not None for w in context.warnings)


def test_detects_deterministic_value_mapping():
    import pandas as pd

    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"], "amount": [1, 2, 3, 4]})
    context = inspect(frame)
    assert any(r.kind == "functional_dependency" for r in context.relationships)
