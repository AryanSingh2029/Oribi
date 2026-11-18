"""Regressions for export privacy and evidence-backed analytical claims."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from oribi import Evidence, ProfileOptions, Warning, inspect


def test_shareable_exports_omit_sensitive_values_by_default():
    frame = pd.DataFrame({
        "a": ["Alice", "Bob"],
        "b": ["alice@example.org", "bob@example.org"],
        "c": ["HIV", "cancer"],
    })
    context = inspect(frame)
    secrets = ("Alice", "Bob", "alice@example.org", "bob@example.org", "HIV", "cancer")
    for exported in (context.to_json(), json.dumps(context.to_dict()), context.to_markdown(), context.to_context()):
        assert not any(value in exported for value in secrets)
    assert all(value in context.to_json(include_values=True) for value in secrets)
    assert all(value in context.to_markdown(include_values=True) for value in secrets)
    assert context.columns[0].statistics.top_values  # Local snapshot retains facts.


def test_exact_bounds_and_tiny_numeric_summaries_require_opt_in():
    frame = pd.DataFrame({
        "reading": [812345.0, 912345.0],
        "date": pd.to_datetime(["2026-01-03", "2026-02-04"]),
    })
    context = inspect(frame)
    safe = context.to_json()
    assert "812345" not in safe and "912345" not in safe
    assert "2026-01-03" not in safe and "2026-02-04" not in safe
    assert context.to_dict()["columns"][0]["statistics"]["mean"] is None
    opt_in = context.to_json(include_values=True)
    assert "812345" in opt_in and "2026-01-03" in opt_in


def test_nested_evidence_strings_cannot_bypass_export_redaction():
    context = inspect(pd.DataFrame({"ordinary": ["a", "b"]}))
    warning = Warning("example", "low", "Example", "No raw values in message", (
        Evidence("nested", "Structured evidence", {"metadata": {"sample": "secret-token", "count": 2}}),
    ))
    context = replace(context, warnings=(warning,))
    assert "secret-token" not in context.to_json()
    assert "secret-token" in context.to_json(include_values=True)
    assert context.to_dict()["warnings"][0]["evidence"][0]["details"]["metadata"]["count"] == 2


def test_near_unique_pair_is_diagnostic_not_observation_grain():
    frame = pd.DataFrame({
        "order_id": [i // 2 for i in range(100)],
        "product_id": [i % 2 for i in range(90)] + [0] * 10,
        "revenue": [float(i) for i in range(100)],
    })
    context = inspect(frame)
    pair = next(key for key in context.candidate_keys if key.columns == ("order_id", "product_id"))
    assert (pair.uniqueness_rate, pair.duplicate_count, pair.missing_count, pair.verified_unique) == (0.95, 5, 0, False)
    assert context.grain.status == "undetermined"
    assert "jointly identify" not in context.to_context()


def test_near_unique_single_column_is_not_rendered_as_identifying_in_markdown():
    context = inspect(pd.DataFrame({"record_key": [f"k{i}" for i in range(99)] + ["k0"]}))
    rendered = context.to_markdown()
    assert "mostly-unique observation-key candidate" in rendered
    assert "record_key** appears to identify observations" not in rendered


def test_missing_key_components_are_not_verified():
    context = inspect(pd.DataFrame({"order_id": [1, 1, 2, 2], "item": [None, "a", None, "b"]}))
    pair = next(key for key in context.candidate_keys if key.kind == "composite")
    assert pair.missing_count == 2
    assert not pair.verified_unique
    assert context.grain.status == "undetermined"


def test_repeated_amount_is_not_an_entity_group():
    context = inspect(pd.DataFrame({"record_id": range(100), "amount": [10.0, 20.0] * 50}))
    assert next(col for col in context.columns if col.name == "amount").semantic.role == "measure"
    assert not any(rel.source_column == "amount" and rel.kind == "one_to_many" for rel in context.relationships)
    assert "amount groups observations" not in context.to_context()


def test_known_mapping_contradiction_is_approximate_only():
    source = [f"s{i}" for i in range(50) for _ in range(2)]
    target = source.copy()
    target[-1] = source[0]
    context = inspect(pd.DataFrame({"source": source, "target": target}))
    assert not any(rel.kind in {"functional_dependency", "one_to_one"} for rel in context.relationships)
    approximate = [rel for rel in context.relationships if rel.kind == "approximate_mapping"]
    assert approximate
    assert all((rel.violation_count, rel.affected_groups, rel.coverage) == (1, 1, 0.98) for rel in approximate)


def test_mapping_with_null_target_is_not_an_exact_dependency():
    source = [f"s{i}" for i in range(50) for _ in range(2)]
    target = source.copy()
    target[-1] = None
    relations = inspect(pd.DataFrame({"source": source, "target": target})).relationships
    assert not any(rel.kind == "functional_dependency" for rel in relations)
    assert any(rel.kind == "approximate_mapping" and rel.missing_pair_count == 1 for rel in relations)


def test_many_to_one_dependency_is_detected_in_the_correct_direction():
    frame = pd.DataFrame({
        "product": [f"p{i // 2}" for i in range(40)],
        "category": [f"c{i // 10}" for i in range(40)],
    })
    relations = inspect(frame).relationships
    assert any(rel.kind == "functional_dependency" and rel.source_column == "product" and rel.target_column == "category" for rel in relations)
    assert not any(rel.kind == "functional_dependency" and rel.source_column == "category" and rel.target_column == "product" for rel in relations)


@pytest.mark.parametrize("seed", range(5))
def test_randomized_exact_mapping_never_has_observed_contradictions(seed):
    import random

    rng = random.Random(seed)
    source = [f"g{i % 15}" for i in range(150)]
    target = [f"v{int(value[1:])}" for value in source]
    for _ in range(seed):
        target[rng.randrange(len(target))] = "conflict"
    frame = pd.DataFrame({"source": source, "target": target})
    relations = inspect(frame).relationships
    for relation in relations:
        if relation.kind == "functional_dependency":
            pair_counts = frame[[relation.source_column, relation.target_column]].groupby(relation.source_column)[relation.target_column].nunique()
            assert pair_counts.max() == 1
            assert relation.violation_count == 0


@pytest.mark.parametrize("width", [10, 40, 100, 200])
def test_wide_context_retains_schema_and_bounds_relationships(width):
    frame = pd.DataFrame({f"category_{i}": [f"v{(row + i) % 10}" for row in range(100)] for i in range(width)})
    context = inspect(frame)
    output = context.to_context()
    assert len(context.relationships) <= 40
    assert len(output) <= 8000
    assert "COLUMNS" in output
    assert all(f"- category_{i}:" in output for i in range(width))
    if width >= 40:
        assert "[Context truncated.]" in output


def test_long_temporal_gap_has_bounded_evidence():
    context = inspect(pd.DataFrame({"date": pd.to_datetime([
        "1900-01-01", "1900-01-02", "1900-01-03", "2000-01-01",
    ])}), options=ProfileOptions(max_gap_examples=2))
    evidence = context.columns[0].warnings[0].evidence[0].details
    assert evidence["missing_intervals"] > 36_000
    assert len(evidence["expected_timestamps"]) == 2
    assert evidence["examples_omitted"] == evidence["missing_intervals"] - 2
    assert len(context.to_json()) < 20_000


def test_gap_example_limit_can_disable_samples():
    context = inspect(pd.DataFrame({"date": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-10",
    ])}), options=ProfileOptions(max_gap_examples=0))
    assert context.columns[0].warnings[0].evidence[0].details["expected_timestamps"] == ()


def test_low_diversity_id_is_not_high_confidence_entity_identifier():
    context = inspect(pd.DataFrame({"status_id": [1, 2] * 50, "amount": [float(i) for i in range(100)]}))
    status = next(col for col in context.columns if col.name == "status_id")
    assert status.semantic.role != "identifier"
    assert not any(rec.operation == "COUNT DISTINCT" for rec in status.recommendations)


def test_named_and_multiindex_are_exported_as_index_metadata():
    named = inspect(pd.DataFrame({"amount": [1.0, 2.0]}, index=pd.Index([10, 11], name="order_id")))
    assert named.profile.index.names == ("order_id",)
    assert named.profile.index.is_unique
    assert "reset_index()" in named.to_context()
    multi = pd.MultiIndex.from_tuples([("A", 1), ("A", 1), ("B", 2)], names=["group", "period"])
    result = inspect(pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=multi))
    assert result.profile.index.level_count == 2
    assert result.profile.index.duplicate_count == 1
    assert result.to_dict()["profile"]["index"]["names"] == ["group", "period"]
    default = inspect(pd.DataFrame({"value": [1.0, 2.0]}))
    assert default.profile.index is None
