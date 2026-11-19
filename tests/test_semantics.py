from oribi import inspect


def test_roles_have_evidence(sales_frame):
    context = inspect(sales_frame)
    roles = {c.name: c.semantic for c in context.columns}
    assert roles["order_id"].role == "identifier"
    assert roles["revenue"].role == "measure"
    assert roles["order_date"].role == "temporal"
    assert roles["order_id"].evidence


def test_numeric_low_cardinality_can_be_dimension():
    context = inspect(__import__("pandas").DataFrame({"rating": [1, 2, 1, 3, 2]}))
    assert context.columns[0].semantic.role == "dimension"


def test_unique_numeric_business_measure_stays_a_measure():
    import pandas as pd

    context = inspect(pd.DataFrame({"revenue": [10.0, 20.0, 35.0, 80.0]}))
    column = context.columns[0]
    assert column.semantic.role == "measure"
    assert {r.operation for r in column.recommendations} >= {"SUM", "MEAN", "MEDIAN"}


def test_unique_integer_revenue_and_amount_stay_measures():
    import pandas as pd

    for name in ("revenue", "amount"):
        column = inspect(pd.DataFrame({name: [10, 20, 35, 80]})).columns[0]
        assert column.semantic.role == "measure"
        assert "name_measure_hint" in {e.code for e in column.semantic.evidence}


def test_numeric_identifier_name_outweighs_numeric_measure_type():
    import pandas as pd

    context = inspect(pd.DataFrame({"customer_id": [101, 102, 103, 104]}))
    assert context.columns[0].semantic.role == "identifier"


def test_reusable_code_is_eligible_for_relationship_detection():
    import pandas as pd

    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"]})
    context = inspect(frame)
    assert context.columns[0].semantic.role == "identifier"
    assert any(r.source_column == "store_code" and r.target_column == "store_name" for r in context.relationships)


def test_ambiguous_unique_numeric_column_remains_unknown():
    import pandas as pd

    context = inspect(pd.DataFrame({"score": [11, 17, 23, 41]}))
    assert context.columns[0].semantic.role == "unknown"


def test_short_numeric_measure_name_is_enough_for_measure_inference():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [2.0, 4.5, 8.0]}))
    assert context.columns[0].semantic.role == "measure"
