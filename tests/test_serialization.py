import json

from datacontext import inspect


def test_serialization_outputs(sales_frame):
    context = inspect(sales_frame)
    assert json.loads(context.to_json()) == context.to_dict()
    assert "## Columns" in context.to_markdown()
    compact = context.to_context(max_chars=500)
    assert len(compact) <= 500
    assert "DATASET" in compact
