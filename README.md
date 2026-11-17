# Oribi

Oribi profiles a pandas DataFrame and returns a structured, evidence-backed snapshot of its shape, column statistics, likely semantic roles, candidate keys, possible grain, relationships, and analytical warnings. Use it to inspect data before writing an analysis or to give downstream tools a compact description of a dataset.

## Install

```bash
python -m pip install oribi
```

Oribi supports Python 3.10–3.14 and pandas 1.5.3 or newer (below pandas 4).

## Quick start

```python
import pandas as pd
from oribi import inspect

df = pd.DataFrame({
    "order_id": [101, 101, 102, 102],
    "product_id": [7, 8, 7, 8],
    "revenue": [25.0, 40.0, 30.0, -5.0],
})

context = inspect(df)
print(context.summary())
print(context.to_context(max_chars=2_000))
```

`inspect()` does not modify the DataFrame or keep a reference to it in the result. The returned snapshot can also be rendered as a full Markdown report or serialized to JSON-compatible data:

```python
markdown = context.to_markdown()
payload = context.to_dict()
json_text = context.to_json()
```

## What Oribi currently reports

- Dataset row/column counts and duplicate rows.
- Per-column missingness, distinct counts, numeric summaries, categorical top values, and datetime bounds.
- Likely roles such as identifier, measure, dimension, categorical, temporal, boolean, text, or unknown, with evidence.
- Candidate single-column and bounded two-column keys, plus a separate likely-grain estimate.
- Conservative within-DataFrame functional-dependency relationships.
- Recommendations for likely numeric measures and identifier counts.
- Warnings for duplicate rows, missingness, non-unique identifiers, negative or skewed measures, high-cardinality categories, and possible gaps in regular datetime series.

Inferences are heuristics, not guarantees about business meaning. Temporal-gap warnings require a sufficiently long series with a dominant fixed cadence; irregular event timestamps and short series are left without a gap claim. Relationships describe observed value mappings within one frame and do not prove that a join is valid or safe.

## Public API

The main operation is `oribi.inspect(df, *, options=None)`. It returns a `DatasetContext` snapshot. `ProfileOptions` can bound top values and exact key/quantile analysis:

```python
import pandas as pd
from oribi import ProfileOptions, inspect

df = pd.DataFrame({"amount": [12.5, 18.0, 20.0]})
context = inspect(df, options=ProfileOptions(top_k=5, max_exact_rows=50_000))
```

Options include `top_k`, `include_quantiles`, `max_exact_rows`, and `max_key_columns`. Column labels must be unique strings. Datetime-like text is not parsed automatically. Profiling is in-memory and may require several full-column scans; exact composite-key checks are bounded by the configured row and column limits.

Confidence values describe heuristic strength; they are not calibrated probabilities. A candidate key is evidence of observed uniqueness under the profiling rules, not proof of a business key. Serialized output includes `schema_version`; treat changes to that output as version-sensitive until a stable compatibility policy is declared.

## Development

```bash
git clone https://github.com/samhoon000/Oribi.git
cd Oribi
python -m venv .venv
```

Activate the environment, then install the package and development tools:

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and [DESIGN.md](DESIGN.md) for architecture notes. Run the end-to-end example from the repository root with `python -m examples.quickstart`.

## Releases

The package version has one source of truth: `project.version` in `pyproject.toml`. To release, update that value and the relevant entry in `CHANGELOG.md`, commit and merge the change, then create and publish a GitHub Release whose tag is exactly `v<version>`. The release workflow reruns tests on Python 3.10–3.14, builds and validates a wheel and source distribution, then publishes those same artifacts to PyPI through GitHub Actions Trusted Publishing. A mismatch or any failed check stops publication. See [CONTRIBUTING.md](CONTRIBUTING.md) for the exact commands and repository setup.

## Planned work

Additional data sources, configurable semantic rules, cross-table foreign-key analysis, and richer profiling options are future work. The current release focuses on pandas DataFrames and deterministic, evidence-bearing heuristics.

## License

Oribi is released under the [MIT License](LICENSE).
