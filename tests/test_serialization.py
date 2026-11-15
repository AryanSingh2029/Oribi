import json

from oribi import inspect


def test_serialization_outputs(sales_frame):
    context = inspect(sales_frame)
    assert json.loads(context.to_json()) == context.to_dict()
    assert "## Columns" in context.to_markdown()
    compact = context.to_context(max_chars=500)
    assert len(compact) <= 500
    assert "DATASET" in compact


def test_result_mappings_cannot_be_mutated_through_the_snapshot(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    try:
        revenue.statistics.quantiles["0.5"] = -100
    except TypeError:
        pass
    else:
        raise AssertionError("quantiles should be read-only")
    assert context.to_dict()["columns"][2]["statistics"]["quantiles"]["0.5"] == 6.25


def test_context_limit_keeps_an_explicit_marker_for_long_labels():
    import pandas as pd

    context = inspect(pd.DataFrame({"x" * 1_000: [1, 2, 3]}))
    rendered = context.to_context(max_chars=100)
    assert len(rendered) <= 100
    assert "[Context truncated.]" in rendered
