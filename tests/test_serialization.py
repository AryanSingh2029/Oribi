import json

from oribi import inspect
from oribi.serialization.context import _format_aggregations, _format_range_value


def test_serialization_outputs(sales_frame):
    context = inspect(sales_frame)
    assert json.loads(context.to_json()) == context.to_dict()
    assert "## Columns" in context.to_markdown()
    compact = context.to_context(max_chars=500)
    assert len(compact) <= 500
    assert "DATASET" in compact


def test_context_renderers_use_scannable_hierarchy_and_compact_columns(sales_frame):
    context = inspect(sales_frame)
    compact = context.to_context()
    assert compact.index("DATASET\n") < compact.index("GRAIN") < compact.index("COLUMNS")
    assert "Missing " in compact
    assert "Unique " in compact
    assert "Aggregations:" in compact
    assert "→" in compact
    assert "Evidence:" not in compact
    assert "confidence" in compact
    assert "DATASET SUMMARY" not in compact
    assert "- order_date" not in compact
    assert " · " in compact
    assert "Missing 25.0%" not in compact

    markdown = context.to_markdown()
    assert markdown.index("## Dataset summary") < markdown.index("## Grain") < markdown.index("## Columns")
    assert "Recommended aggregations:" in markdown
    assert "<details>" in markdown
    assert "<summary>Inference evidence</summary>" in markdown


def test_compact_context_formats_column_warnings_concisely():
    import pandas as pd

    context = inspect(pd.DataFrame({"region": ["East", "West", "East", None]}))
    rendered = context.to_context()
    assert "⚠ region · 25% missing" in rendered
    assert "[medium]" not in rendered


def test_aggregation_formatter_preserves_names_as_complete_values():
    assert _format_aggregations(["COUNT DISTINCT"]) == "COUNT DISTINCT"
    assert _format_aggregations(["SUM", "MEAN"]) == "SUM · MEAN"
    assert _format_aggregations([]) == "—"
    assert _format_aggregations(["SUM, MEAN", "COUNT DISTINCT"]) == "SUM, MEAN · COUNT DISTINCT"
    assert _format_aggregations("COUNT DISTINCT") == "COUNT DISTINCT"


def test_range_formatter_removes_unnecessary_time_and_decimal_parts():
    import pandas as pd

    assert _format_range_value(pd.Timestamp("2026-01-01 00:00:00")) == "2026-01-01"
    assert _format_range_value(pd.Timestamp("2026-01-01 12:30:00")) == "2026-01-01 12:30:00"
    assert _format_range_value(85.0) == "85"
    assert _format_range_value(85.5) == "85.5"


def test_compact_context_formats_date_and_whole_number_ranges():
    import pandas as pd

    context = inspect(pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "amount": [85.0, 210.0],
    }))
    rendered = context.to_context()
    assert "Range 2026-01-01 → 2026-01-02" in rendered
    assert "Range 85 → 210" in rendered


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
