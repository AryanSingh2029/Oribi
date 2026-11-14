import pandas as pd

from datacontext import inspect


def test_composite_key_and_grain(sales_frame):
    context = inspect(sales_frame)
    assert context.grain is not None
    assert context.grain.status == "estimated"
    assert len(context.grain.columns) >= 1
    assert context.candidate_keys


def test_uncertain_grain_is_allowed():
    context = inspect(pd.DataFrame({"group": ["a", "a", "a"], "value": [1, 1, 1]}))
    assert context.grain.status == "undetermined"
