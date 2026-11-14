from datacontext import inspect


def test_roles_have_evidence(sales_frame):
    context = inspect(sales_frame)
    roles = {c.name: c.semantic for c in context.columns}
    assert roles["order_id"].role == "identifier"
    assert roles["revenue"].role == "measure"
    assert roles["order_date"].role == "temporal"
    assert roles["order_id"].evidence


def test_numeric_low_cardinality_can_be_dimension():
    context = inspect(__import__("pandas").DataFrame({"rating": [1, 2, 1, 3, 2]}))
    assert context.columns[0].semantic.role in {"dimension", "measure"}
